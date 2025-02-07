import os
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from ..tool import PydanticTool


class WriteFileParams(BaseModel):
    path: str = Field(description="Relative path where to write the file")
    content: str = Field(description="Content to write to the file")

    @field_validator("path")
    def validate_path(cls, path: str) -> str:
        # Ensure path doesn't contain parent directory traversal
        if ".." in Path(path).parts:
            raise ValueError("Path cannot contain parent directory traversal (..)")

        # Convert to posix path format and ensure it's relative
        clean_path = Path(path).as_posix()
        if os.path.isabs(clean_path):
            raise ValueError("Path must be relative")

        return clean_path


class WriteFileTool(PydanticTool[WriteFileParams]):
    name = "write_file"
    description = "Write content to a file at the specified relative path"
    Params = WriteFileParams

    def __init__(self, root: str | Path):
        """Initialize tool with root directory path."""
        self.root = Path(root).resolve()

        # Ensure root directory exists
        if not self.root.exists():
            raise ValueError(f"Root directory does not exist: {self.root}")
        if not self.root.is_dir():
            raise ValueError(f"Root path is not a directory: {self.root}")

    def serialize_params(self) -> dict:
        return {"root": self.root}

    def call(self, params: WriteFileParams) -> str:
        # Combine and resolve the full path
        full_path = (self.root / params.path).resolve()

        # Verify the resolved path is within root directory
        try:
            full_path.relative_to(self.root)
        except ValueError:
            raise ValueError(f"Path {params.path} resolves outside root directory")

        # Create parent directories if they don't exist
        full_path.parent.mkdir(parents=True, exist_ok=True)

        # Write the file
        full_path.write_text(params.content)

        return f"Successfully wrote {len(params.content)} characters to {params.path}"
