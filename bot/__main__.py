"""Allow ``python -m bot`` as a shortcut for ``python -m bot.main``."""
from .main import main

if __name__ == "__main__":
    main()
