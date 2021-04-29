from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
import orjson
import os
import aiofiles
import contextlib


NUM_OVERLAPS = int(os.environ.get("NUM_OVERLAPS", 10))
ALIGNMENT_MAX_SIZE = int(os.environ.get("ALIGNMENT_MAX_SIZE", 8))

CMD_OVERLAPS = "/apps/vecalign/overlap.py"
CMD_EMBED = "/apps/LASER/tasks/embed/embed.sh"
CMD_ALIGN = "/apps/vecalign/vecalign.py"

SOURCE_TEXT_FILENAME = "/tmp/source_text"
TARGET_TEXT_FILENAME = "/tmp/target_text"
SOURCE_OVERLAPS_FILENAME = "/tmp/source_overlaps"
TARGET_OVERLAPS_FILENAME = "/tmp/target_overlaps"
SOURCE_EMBEDING_FILENAME = "/tmp/source_embeding"
TARGET_EMBEDING_FILENAME = "/tmp/target_embeding"


class InputData(BaseModel):
    source_text: str
    source_language: str
    target_text: str
    target_language: str

    class Config:
        schema_extra = {
            "example": {
                "source_text": "Action taken on Parliament's resolutions: see Minutes Documents received: see Minutes\nWritten statements (Rule 116): see Minutes\nTexts of agreements forwarded by the Council: see Minutes\nMembership of Parliament: see Minutes\nMembership of committees and delegations: see Minutes\nFuture action in the field of patents (motions for resolutions tabled): see Minutes\nAgenda for next sitting: see Minutes\nClosure of sitting",
                "source_language": "en",
                "target_text": "Následný postup na základě usnesení Parlamentu: viz zápis\nPředložení dokumentů: viz zápis\nPísemná prohlášení (článek 116 jednacího řádu): viz zápis\nTexty smluv dodané Radou: viz zápis\nSložení Parlamentu: viz zápis\nČlenství ve výborech a delegacích: viz zápis\nBudoucí akce v oblasti patentů (předložené návrhy usnesení): viz zápis Pořad jednání příštího zasedání: viz zápis\nUkončení zasedání",
                "target_language": "cs",
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
                    "Future action in the field of patents (motions for resolutions tabled): see Minutes Agenda for next sitting: see Minutes",
                    "Closure of sitting",
                ],
                "target": [
                    "Následný postup na základě usnesení Parlamentu: viz zápis Předložení dokumentů: viz zápis",
                    "Písemná prohlášení (článek 116 jednacího řádu): viz zápis",
                    "Texty smluv dodané Radou: viz zápis",
                    "Složení Parlamentu: viz zápis",
                    "Členství ve výborech a delegacích: viz zápis",
                    "Budoucí akce v oblasti patentů (předložené návrhy usnesení): viz zápis Pořad jednání příštího zasedání: viz zápis",
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


class Message500(BaseModel):
    class Config:
        schema_extra = {
            "example": {
                "detail": "Processing failed",
            },
        }


class OverlapsException(Exception):
    pass


class EmbedingException(Exception):
    pass


class AlignmentException(Exception):
    pass


async def run_cmd_async(cmd: list[str]) -> tuple[str, int]:
    process = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE)

    data = await process.stdout.read()
    code = await process.wait()
    return data.decode().strip(), code


async def create_input_file(text_filename: str, text: str):
    async with aiofiles.open(text_filename, mode="w") as f:
        await f.write(text)


async def create_overlaps(text_filename: str, overlaps_filename: str):
    # ./overlap.py -i bleualign_data/dev.fr bleualign_data/test*.fr -o bleualign_data/overlaps.fr -n 10
    output, exit_code = await run_cmd_async(
        [
            CMD_OVERLAPS,
            "-i",
            text_filename,
            "-o",
            overlaps_filename,
            "-n",
            str(NUM_OVERLAPS),
        ]
    )
    if exit_code != 0:
        raise OverlapsException(output)


async def create_embeding(
    overlaps_filename: str, language: str, embeding_filename: str
):
    # $LASER/tasks/embed/embed.sh bleualign_data/overlaps.fr fr bleualign_data/overlaps.fr.emb
    output, exit_code = await run_cmd_async(
        [CMD_EMBED, overlaps_filename, language, embeding_filename]
    )
    if exit_code != 0:
        raise EmbedingException(output)


async def create_alignment(
    source_text_filename: str,
    source_overlaps_filename: str,
    source_embeding_filename: str,
    target_text_filename: str,
    target_overlaps_filename: str,
    target_embeding_filename: str,
) -> str:
    alignment, exit_code = await run_cmd_async(
        [
            CMD_ALIGN,
            "--alignment_max_size",
            str(ALIGNMENT_MAX_SIZE),
            "--src",
            source_text_filename,
            "--tgt",
            target_text_filename,
            "--src_embed",
            source_overlaps_filename,
            source_embeding_filename,
            "--tgt_embed",
            target_overlaps_filename,
            target_embeding_filename,
        ]
    )
    if exit_code != 0:
        raise AlignmentException
    return alignment


def parse_alignment_line(line: str) -> tuple:
    source_indices, target_indices, cost = line.split(":")
    return orjson.loads(source_indices), orjson.loads(target_indices), float(cost)


app = FastAPI()


@app.post(
    "/",
    response_model=OutputData,
    responses={
        200: {"model": OutputData},
        500: {"model": Message500},
    },
)
async def align_text(data: InputData):
    try:
        await create_input_file(SOURCE_TEXT_FILENAME, data.source_text)
        await create_input_file(TARGET_TEXT_FILENAME, data.target_text)

        # overlaps
        await create_overlaps(SOURCE_TEXT_FILENAME, SOURCE_OVERLAPS_FILENAME)
        await create_overlaps(TARGET_TEXT_FILENAME, TARGET_OVERLAPS_FILENAME)

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
        )

        # format output
        pairing = [parse_alignment_line(line) for line in alignment.splitlines()]
        source_lines = data.source_text.splitlines()
        target_lines = data.target_text.splitlines()
        source_aligned = []
        target_aligned = []
        for source_indices, target_indices, _ in pairing:
            source_aligned.append(
                " ".join(source_lines[index] for index in source_indices)
            )
            target_aligned.append(
                " ".join(target_lines[index] for index in target_indices)
            )

        return OutputData(
            source=source_aligned,
            target=target_aligned,
            pairing=pairing,
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
