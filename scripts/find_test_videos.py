
import asyncio
from sqlalchemy import select, desc
from hbmon import db, models

async def find_test_videos():
    await db.init_async_db()
    async with db.async_session_scope() as session:
        # Find True Positive (High Confidence)
        print("--- Candidates for True Positive (TP) ---")
        stmt_tp = (
            select(models.Observation)
            .order_by(desc(models.Observation.species_prob))
            .limit(5)
        )
        result_tp = await session.execute(stmt_tp)
        for obs in result_tp.scalars():
            print(f"TP: ID={obs.id}, Species={obs.species_label}, Prob={obs.species_prob:.4f}, Video={obs.video_path}")

        # Find False Positive / Rejected (Candidate table)
        print("\n--- Candidates for False Positive / Rejects ---")
        # We need to query the Candidates table for rejected items
        # Note: 'models.Candidate' needs to be available. 
        # If it's not exposed in models.py __all__, we might need to look at worker.py or raw SQL.
        # Checking models.py content previously shown... Candidate IS in __all__.
        
        # Candidate table contains rejected items. 
        # Reject reason is stored in `extra_json`
        stmt_fp = (
            select(models.Candidate)
            .order_by(desc(models.Candidate.ts))
            .limit(10)
        )
        try:
            result_fp = await session.execute(stmt_fp)
            for cand in result_fp.scalars():
                extra = cand.get_extra() or {}
                # Only interested if it has a video
                if not cand.clip_path:
                    continue
                    
                status = extra.get("status", "unknown")
                reason = extra.get("reject_reason", "unknown")
                print(f"FP/Reject: ID={cand.id}, Status={status}, Reason={reason}, Video={cand.clip_path}")
        except Exception as e:
            print(f"Error querying candidates: {e}")
            
if __name__ == "__main__":
    # Rely on environment being configured correctly (via env vars or defaults in hbmon/config.py)
    asyncio.run(find_test_videos())
