import click
import multiprocessing
from app import run_app


@click.command()
@click.option("--env", default="primary", show_default=True)
@click.option("--debug", is_flag=True, default=False, show_default=True)
def main(env, debug):
    run_app(env=env, debug=debug, no_processor=debug)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
