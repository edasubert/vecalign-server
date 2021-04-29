import orjson
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import contextlib

from exceptions import (
    Message500,
    OverlapsException,
    EmbedingException,
    AlignmentException,
)
from common import (
    write_file,
    read_file,
    create_overlaps,
    create_embeding,
    create_alignment,
    format_output,
)


NUM_OVERLAPS = int(os.environ.get("NUM_OVERLAPS", 10))
ALIGNMENT_MAX_SIZE = int(os.environ.get("ALIGNMENT_MAX_SIZE", 8))

SOURCE_TEXT_FILENAME = "/tmp/source_text"
TARGET_TEXT_FILENAME = "/tmp/target_text"
SOURCE_OVERLAPS_FILENAME = "/tmp/source_overlaps"
TARGET_OVERLAPS_FILENAME = "/tmp/target_overlaps"
SOURCE_EMBEDING_FILENAME = "/tmp/source_embeding"
TARGET_EMBEDING_FILENAME = "/tmp/target_embeding"


class InputDataText(BaseModel):
    source_language: str
    source_text: str
    target_language: str
    target_text: str

    class Config:
        schema_extra = {
            "example": {
                "source_text": """Action taken on Parliament's resolutions: see Minutes Documents received: see Minutes\n
                Written statements (Rule 116): see Minutes\n
                Texts of agreements forwarded by the Council: see Minutes\n
                Membership of Parliament: see Minutes\n
                Membership of committees and delegations: see Minutes\n
                Future action in the field of patents (motions for resolutions tabled): see Minutes\n
                Agenda for next sitting: see Minutes\n
                Closure of sitting""",
                "source_language": "en",
                "target_text": """Následný postup na základě usnesení Parlamentu: viz zápis\n
                Předložení dokumentů: viz zápis\n
                Písemná prohlášení (článek 116 jednacího řádu): viz zápis\n
                Texty smluv dodané Radou: viz zápis\n
                Složení Parlamentu: viz zápis\n
                Členství ve výborech a delegacích: viz zápis\n
                Budoucí akce v oblasti patentů (předložené návrhy usnesení):
                viz zápis Pořad jednání příštího zasedání: viz zápis\n
                Ukončení zasedání""",
                "target_language": "cs",
            },
        }


class InputDataFiles(BaseModel):
    source_language: str
    source_text_filename: str
    source_overlaps_filename: str
    source_embeding_filename: str
    target_language: str
    target_text_filename: str
    target_overlaps_filename: str
    target_embeding_filename: str
    alignment_filename: str

    class Config:
        schema_extra = {
            "example": {
                "source_language": "en",
                "source_text_filename": "/data/en",
                "source_overlaps_filename": "/data/en.overlaps",
                "source_embeding_filename": "/data/en.emb",
                "target_language": "cs",
                "target_text_filename": "/data/cs",
                "target_overlaps_filename": "/data/cs.overlaps",
                "target_embeding_filename": "/data/cs.emb",
                "alignment_filename": "/data/cs.alignment",
            },
        }


class OutputData(BaseModel):
    source: list[str]
    target: list[str]
    pairing: list[tuple[list, list, float]]

    class Config:
        schema_extra = {
            "example": {
                "source": [
                    "Action taken on Parliament's resolutions: see Minutes Documents received: see Minutes",
                    "Written statements (Rule 116): see Minutes",
                    "Texts of agreements forwarded by the Council: see Minutes",
                    "Membership of Parliament: see Minutes",
                    "Membership of committees and delegations: see Minutes",
                    "Future action in the field of patents (motions for resolutions tabled): "
                    "see Minutes Agenda for next sitting: see Minutes",
                    "Closure of sitting",
                ],
                "target": [
                    "Následný postup na základě usnesení Parlamentu: viz zápis Předložení dokumentů: viz zápis",
                    "Písemná prohlášení (článek 116 jednacího řádu): viz zápis",
                    "Texty smluv dodané Radou: viz zápis",
                    "Složení Parlamentu: viz zápis",
                    "Členství ve výborech a delegacích: viz zápis",
                    "Budoucí akce v oblasti patentů (předložené návrhy usnesení): viz "
                    "zápis Pořad jednání příštího zasedání: viz zápis",
                    "Ukončení zasedání",
                ],
                "pairing": [
                    [[0], [0, 1], 0.10611],
                    [[1], [2], 0.041514],
                    [[2], [3], 0.103042],
                    [[3], [4], 0.114679],
                    [[4], [5], 0.059114],
                    [[5, 6], [6], 0.044997],
                    [[7], [7], 0.227794],
                ],
            },
        }


app = FastAPI()


@app.post(
    "/align_text",
    response_model=OutputData,
    responses={
        200: {"model": OutputData},
        500: {"model": Message500},
    },
)
async def align_text(data: InputDataText):
    try:
        await write_file(SOURCE_TEXT_FILENAME, data.source_text)
        await write_file(TARGET_TEXT_FILENAME, data.target_text)

        # overlaps
        await create_overlaps(
            SOURCE_TEXT_FILENAME, SOURCE_OVERLAPS_FILENAME, NUM_OVERLAPS
        )
        await create_overlaps(
            TARGET_TEXT_FILENAME, TARGET_OVERLAPS_FILENAME, NUM_OVERLAPS
        )

        # embeding
        await create_embeding(
            SOURCE_OVERLAPS_FILENAME, data.source_language, SOURCE_EMBEDING_FILENAME
        )
        await create_embeding(
            TARGET_OVERLAPS_FILENAME, data.target_language, TARGET_EMBEDING_FILENAME
        )

        # alignment
        alignment = await create_alignment(
            SOURCE_TEXT_FILENAME,
            SOURCE_OVERLAPS_FILENAME,
            SOURCE_EMBEDING_FILENAME,
            TARGET_TEXT_FILENAME,
            TARGET_OVERLAPS_FILENAME,
            TARGET_EMBEDING_FILENAME,
            ALIGNMENT_MAX_SIZE,
        )

        return OutputData(
            **format_output(alignment, data.source_text, data.target_text)
        )
    except (OverlapsException, EmbedingException, AlignmentException) as e:
        raise HTTPException(
            status_code=500, detail="Processing Failed: " + str(e)
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail="Unexpected error during processing: " + str(e)
        ) from e
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.remove(SOURCE_TEXT_FILENAME)
        with contextlib.suppress(FileNotFoundError):
            os.remove(TARGET_TEXT_FILENAME)
        with contextlib.suppress(FileNotFoundError):
            os.remove(SOURCE_OVERLAPS_FILENAME)
        with contextlib.suppress(FileNotFoundError):
            os.remove(TARGET_OVERLAPS_FILENAME)
        with contextlib.suppress(FileNotFoundError):
            os.remove(SOURCE_EMBEDING_FILENAME)
        with contextlib.suppress(FileNotFoundError):
            os.remove(TARGET_EMBEDING_FILENAME)


@app.post(
    "/align_files_in_place",
    responses={
        200: {},
        500: {"model": Message500},
    },
)
async def align_files_in_place(data: InputDataFiles):
    try:
        # overlaps
        await create_overlaps(
            data.source_text_filename, data.source_overlaps_filename, NUM_OVERLAPS
        )
        await create_overlaps(
            data.target_text_filename, data.target_overlaps_filename, NUM_OVERLAPS
        )

        # embeding
        await create_embeding(
            data.source_overlaps_filename,
            data.source_language,
            data.source_embeding_filename,
        )
        await create_embeding(
            data.target_overlaps_filename,
            data.target_language,
            data.target_embeding_filename,
        )

        # alignment
        alignment = await create_alignment(
            data.source_text_filename,
            data.source_overlaps_filename,
            data.source_embeding_filename,
            data.target_text_filename,
            data.target_overlaps_filename,
            data.target_embeding_filename,
            ALIGNMENT_MAX_SIZE,
        )

        await write_file(
            data.alignment_filename,
            orjson.dumps(
                format_output(
                    alignment,
                    await read_file(data.source_text_filename),
                    await read_file(data.target_text_filename),
                )
            ).decode(),
        )
    except (OverlapsException, EmbedingException, AlignmentException) as e:
        raise HTTPException(
            status_code=500, detail="Processing Failed: " + str(e)
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail="Unexpected error during processing: " + str(e)
        ) from e
