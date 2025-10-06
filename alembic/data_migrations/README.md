# Data Migration Scripts

This directory contains data migration scripts for backfilling and transforming existing data when schema migrations add new fields.

## Available Scripts

### 1. Backfill Context Fields (`backfill_context_fields.py`)

**Purpose:** Backfills `context_pruned`, `pruning_count`, and `total_context_tokens` fields for existing calls.

**When to run:** After applying migration `003_add_context_management_fields`

**Usage:**
```bash
python -m alembic.data_migrations.backfill_context_fields
```

**What it does:**
- Finds all calls with default context field values (context_pruned=false, pruning_count=0, total_context_tokens=0)
- Analyzes conversation history for each call
- Estimates token counts based on message content length (~4 chars per token)
- Estimates if context was pruned based on message count and token count
- Updates calls with calculated values

**Configuration:**
- Batch size: 100 calls per batch (configurable in code)
- Only processes calls older than 1 hour to avoid active calls

**Example output:**
```
INFO: Found 1523 calls to backfill
INFO: Updated 100 calls (batch offset 0)
INFO: Progress: 100/1523 (6.6%)
INFO: Updated 100 calls (batch offset 100)
INFO: Progress: 200/1523 (13.1%)
...
INFO: Backfill complete! Updated 1523 calls
```

---

### 2. Backfill Message Tokens (`backfill_message_tokens.py`)

**Purpose:** Backfills `token_count` field for existing conversation history messages.

**When to run:** After applying migration `003_add_context_management_fields`

**Usage:**
```bash
python -m alembic.data_migrations.backfill_message_tokens
```

**What it does:**
- Finds all messages with token_count=0 and non-empty content
- Estimates token count using ~4 characters per token heuristic
- Updates messages with calculated token counts

**Configuration:**
- Batch size: 500 messages per batch (configurable in code)

**Example output:**
```
INFO: Found 12456 messages to backfill
INFO: Updated 500 messages (batch offset 0)
INFO: Progress: 500/12456 (4.0%)
INFO: Updated 500 messages (batch offset 500)
INFO: Progress: 1000/12456 (8.0%)
...
INFO: Backfill complete! Updated 12456 messages
```

---

## Running Data Migrations

### Prerequisites

1. Ensure schema migrations are applied first:
```bash
alembic upgrade head
```

2. Set database credentials (if not using settings.py):
```bash
export DATABASE_URL="postgresql+asyncpg://user:password@localhost:5432/dbname"
```

### Recommended Execution Order

After applying migration `003_add_context_management_fields`:

```bash
# 1. Backfill message token counts first (faster, independent)
python -m alembic.data_migrations.backfill_message_tokens

# 2. Then backfill call context fields (depends on message data)
python -m alembic.data_migrations.backfill_context_fields
```

### Production Considerations

**Safety:**
- Scripts use batched updates to avoid long-running transactions
- Progress is logged for monitoring
- Scripts are idempotent (safe to re-run)

**Performance:**
- Run during low-traffic periods
- Monitor database load during execution
- Adjust batch sizes if needed:
  ```python
  backfiller = ContextFieldBackfiller(database_url, batch_size=50)  # Smaller batches
  ```

**Monitoring:**
- Check logs for progress updates
- Monitor database CPU/memory usage
- Verify results with sample queries:
  ```sql
  -- Check context field backfill
  SELECT
      COUNT(*) as total,
      SUM(CASE WHEN context_pruned THEN 1 ELSE 0 END) as pruned,
      AVG(total_context_tokens) as avg_tokens
  FROM calls
  WHERE created_at < NOW() - INTERVAL '1 hour';

  -- Check message token backfill
  SELECT
      COUNT(*) as total_messages,
      SUM(token_count) as total_tokens,
      AVG(token_count) as avg_tokens_per_message
  FROM conversation_history
  WHERE content IS NOT NULL;
  ```

---

## Creating New Data Migration Scripts

When creating new data migration scripts:

1. **Follow naming convention:** `backfill_<feature_name>.py`

2. **Include docstring with:**
   - Purpose
   - When to run
   - Usage command

3. **Implement key methods:**
   ```python
   class MyBackfiller:
       async def setup(self):
           # Initialize database connection

       async def teardown(self):
           # Clean up resources

       async def count_records_to_backfill(self) -> int:
           # Count records needing backfill

       async def backfill_batch(self, offset: int) -> int:
           # Process one batch

       async def run(self):
           # Main execution logic
   ```

4. **Use batched processing:**
   - Configurable batch size
   - Progress logging
   - Transaction per batch

5. **Make it idempotent:**
   - Safe to re-run
   - Skip already-processed records
   - No duplicate updates

6. **Add to this README** with usage instructions

---

## Troubleshooting

**Script hangs:**
- Check database connection
- Verify no long-running locks
- Reduce batch size

**Incorrect estimates:**
- Token counts are estimates (~4 chars/token)
- For production accuracy, use actual tokenizer (tiktoken/transformers)
- Estimates are adequate for monitoring/analytics

**Partial completion:**
- Scripts are idempotent - safe to re-run
- Will resume from where it stopped
- Check logs for last successful batch

**Database connection errors:**
- Verify DATABASE_URL or settings.py configuration
- Check database permissions
- Ensure database is accessible from current host
