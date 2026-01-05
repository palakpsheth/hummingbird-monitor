import asyncio
import os
import sys

# Ensure src is in path
sys.path.append(os.path.abspath("src"))
sys.path.append("/app/src")

from hbmon.db import get_async_db
from hbmon.models import Observation
from sqlalchemy import select

async def check():
    print("Checking Obs 53 state...")
    async with get_async_db() as db:
        result = await db.execute(select(Observation).where(Observation.id==53))
        obs = result.scalar()
        if obs:
            print(f"Obs 53 ID: {obs.id}")
            print(f"Obs 53 State: '{obs.annotation_state}'")
        else:
            print("Obs 53 NOT FOUND")

if __name__ == "__main__":
    asyncio.run(check())
