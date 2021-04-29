import asyncio
import aiofiles
import orjson

from exceptions import OverlapsException, EmbedingException, AlignmentException


CMD_OVERLAPS = "/apps/vecalign/overlap.py"
CMD_EMBED = "/apps/LASER/tasks/embed/embed.sh"
CMD_ALIGN = "/apps/vecalign/vecalign.py"


async def run_cmd_async(cmd: list[str]) -> tuple[str, int]:
    process = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE)

    data = await process.stdout.read()
    code = await process.wait()
    return data.decode().strip(), code


async def write_file(text_filename: str, text: str):
    async with aiofiles.open(text_filename, mode="w") as f:
        await f.write(text)


async def read_file(text_filename: str) -> str:
    async with aiofiles.open(text_filename, mode="r") as f:
        return await f.read()


async def create_overlaps(
    text_filename: str, overlaps_filename: str, num_overlaps: int
):
    # ./overlap.py -i bleualign_data/dev.fr bleualign_data/test*.fr -o bleualign_data/overlaps.fr -n 10
    output, exit_code = await run_cmd_async(
        [
            CMD_OVERLAPS,
            "-i",
            text_filename,
            "-o",
            overlaps_filename,
            "-n",
            str(num_overlaps),
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
    alignment_max_size: int,
) -> str:
    alignment, exit_code = await run_cmd_async(
        [
            CMD_ALIGN,
            "--alignment_max_size",
            str(alignment_max_size),
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


def format_output(alignment, source_text, target_text):
    pairing = [parse_alignment_line(line) for line in alignment.splitlines()]
    source_lines = source_text.splitlines()
    target_lines = target_text.splitlines()
    source_aligned = []
    target_aligned = []
    for source_indices, target_indices, _ in pairing:
        source_aligned.append(" ".join(source_lines[index] for index in source_indices))
        target_aligned.append(" ".join(target_lines[index] for index in target_indices))
    return {
        "source": source_aligned,
        "target": target_aligned,
        "pairing": pairing,
    }
