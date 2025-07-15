from pathlib import Path
import subprocess
from rich import print
from omni.utils.schemas import RunConfig


def setup(definitions: list, run_config: RunConfig) -> None:
    """
    Setup the environment.

    Args:
        definitions (List[str]): List of images to build.
        run_config (RunConfig): Information related to the run.
    """

    # Create the directory if it doesn't exist
    run_config.images_directory.mkdir(parents=True, exist_ok=True)

    for image in definitions:
        # Check all def files that start with the image name
        def_files = list(Path("omni/definitions").glob(f"{image}*.def"))

        for def_file in def_files:
            # Check if the definition file exists
            if not def_file.exists():
                print(f"[red]Definition file {def_file} does not exist.[/red]")
                continue

            # Check if the image already exists
            image_path = Path(run_config.images_directory) / def_file.name.replace(
                ".def", ""
            )
            if image_path.exists():
                print(
                    f"[green]Image {def_file.name.replace('.def', '')} already exists.[/green]"
                )
                continue

            # Build the image
            subprocess.run(
                [
                    "apptainer",
                    "build",
                    "--sandbox",
                    str(image_path),
                    str(def_file),
                ],
                check=True,
            )

            print(
                f"[green]Image {def_file.name.replace('.def', '')} built successfully.[/green]"
            )
