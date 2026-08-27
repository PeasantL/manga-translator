from dotenv import load_dotenv

# Load .env here rather than in an entry point so that every consumer of the
# package sees the same environment. server.py used to be the only caller, which
# meant a .env file worked for the web UI but silently did nothing for the CLI.
load_dotenv()
