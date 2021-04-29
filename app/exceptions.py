from pydantic import BaseModel


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
