"""Allow running Pengu as: python -m pengu"""

import asyncio

from pengu.app import main

if __name__ == "__main__":
    asyncio.run(main())
