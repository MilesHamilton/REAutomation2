# Database Migration Guide

Complete guide for managing database migrations in the REAutomation project using Alembic.

## Table of Contents

1. [Overview](#overview)
2. [Migration Structure](#migration-structure)
3. [Running Migrations](#running-migrations)
4. [Creating New Migrations](#creating-new-migrations)
5. [Testing Migrations](#testing-migrations)
6. [Data Migrations](#data-migrations)
7. [Troubleshooting](#troubleshooting)
8. [Production Deployment](#production-deployment)

---

## Overview

This project uses **Alembic** for database schema versioning and migrations. All schema changes must go through Alembic migrations to ensure:

- Version control of database schema
- Repeatable deployments
- Rollback capability
- Data preservation during schema changes

**Key Principles:**
- Never modify `src/database/models.py` without creating a corresponding migration
- Always test migrations (upgrade AND downgrade) before deploying
- Include data migrations for backfilling new fields
- Document breaking changes and required actions

---

## Migration Structure

### Migration Files

Located in `alembic/versions/`:

1. **`26a6cb1543c8_add_base_tables.py`** - Base schema
   - Core tables: calls, contacts, conversation_history, call_notes
   - Base indexes for primary queries

2. **`001_add_langsmith_monitoring_tables.py`** - LangSmith monitoring
   - Tables: workflow_traces, agent_executions, performance_metrics
   - System metrics tracking

3. **`002_add_voice_agent_integration_fields.py`** - Voice agent integration
   - Tables: agent_transitions, tier_escalation_events
   - Voice-specific fields and indexes

4. **`003_add_context_management_fields.py`** - Context management
   - Fields: context_pruned, pruning_count, total_context_tokens
   - Message importance and token tracking
   - Indexes for context analytics

5. **`004_add_performance_indexes.py`** - Performance optimization
   - Covering indexes for dashboard queries
   - Partial indexes for selective queries
   - Time-series optimized indexes for metrics

### Data Migration Scripts

Located in `alembic/data_migrations/`:

- `backfill_context_fields.py` - Backfill context management fields for existing calls
- `backfill_message_tokens.py` - Backfill token counts for conversation history
- See `alembic/data_migrations/README.md` for detailed usage

### Migration Tests

Located in `tests/migrations/`:

- `test_migrations.py` - Comprehensive test suite for all migrations
- Tests upgrade/downgrade, data preservation, index effectiveness

---

## Running Migrations

### Development Environment

1. **Check current revision:**
```bash
alembic current
```

2. **View migration history:**
```bash
alembic history --verbose
```

3. **Upgrade to latest:**
```bash
alembic upgrade head
```

4. **Upgrade to specific revision:**
```bash
alembic upgrade 003_context_management
```

5. **Downgrade one revision:**
```bash
alembic downgrade -1
```

6. **Downgrade to specific revision:**
```bash
alembic downgrade 002_voice_agent_integration
```

7. **View SQL without executing:**
```bash
alembic upgrade head --sql
```

### Database Setup from Scratch

```bash
# 1. Ensure database exists
psql -U postgres -c "CREATE DATABASE reautomation;"

# 2. Run all migrations
alembic upgrade head

# 3. Verify tables created
psql -U postgres -d reautomation -c "\dt"

# 4. Run data migrations if needed (for existing data)
python -m alembic.data_migrations.backfill_message_tokens
python -m alembic.data_migrations.backfill_context_fields
```

---

## Creating New Migrations

### Auto-generate Migration from Model Changes

**Best practice:** Alembic can detect changes in `src/database/models.py` and generate migrations automatically.

```bash
# 1. Modify models in src/database/models.py
# 2. Generate migration
alembic revision --autogenerate -m "add_new_field_to_calls"

# 3. Review generated migration in alembic/versions/
# 4. Edit if needed (autogenerate may miss some changes)
# 5. Test migration
alembic upgrade head
alembic downgrade -1
alembic upgrade head
```

### Manual Migration Creation

For complex changes that autogenerate can't detect:

```bash
# Create empty migration
alembic revision -m "add_custom_index"
```

Then edit the generated file:

```python
def upgrade() -> None:
    """Add custom composite index."""
    op.create_index(
        'ix_calls_custom',
        'calls',
        ['status', 'created_at', 'total_cost'],
        postgresql_using='btree'
    )

def downgrade() -> None:
    """Remove custom composite index."""
    op.drop_index('ix_calls_custom', table_name='calls')
```

### Migration Best Practices

**DO:**
- ✅ Test both upgrade AND downgrade
- ✅ Use server defaults for new NOT NULL columns
- ✅ Create data migrations for backfilling
- ✅ Document breaking changes in migration docstring
- ✅ Use descriptive revision messages
- ✅ Add indexes for new query patterns

**DON'T:**
- ❌ Rename columns without data preservation
- ❌ Add NOT NULL columns without defaults
- ❌ Drop columns with data (create archive table first)
- ❌ Skip testing downgrade
- ❌ Modify existing migrations (create new one)

### Adding New Fields

Example: Adding a `priority` field to calls table

**1. Update model:**
```python
# src/database/models.py
class Call(Base):
    # ... existing fields ...
    priority: Mapped[int] = mapped_column(Integer, server_default="0", nullable=False, index=True)
```

**2. Generate migration:**
```bash
alembic revision --autogenerate -m "add_priority_field_to_calls"
```

**3. Review generated migration:**
```python
def upgrade() -> None:
    op.add_column('calls', sa.Column('priority', sa.Integer(), server_default='0', nullable=False))
    op.create_index('ix_calls_priority', 'calls', ['priority'])

def downgrade() -> None:
    op.drop_index('ix_calls_priority', table_name='calls')
    op.drop_column('calls', 'priority')
```

**4. Test migration:**
```bash
alembic upgrade head  # Apply
alembic downgrade -1  # Rollback
alembic upgrade head  # Re-apply
```

**5. Create data migration if needed** (to backfill priority for existing calls)

---

## Testing Migrations

### Running Migration Tests

```bash
# Run all migration tests
pytest tests/migrations/ -v

# Run specific test
pytest tests/migrations/test_migrations.py::test_003_context_management_upgrade -v

# Run with detailed output
pytest tests/migrations/ -v -s
```

### Test Coverage

The test suite (`tests/migrations/test_migrations.py`) includes:

1. **Upgrade Tests** - Verify schema changes applied correctly
2. **Downgrade Tests** - Verify rollback removes changes
3. **Data Preservation Tests** - Verify existing data not lost
4. **Index Effectiveness Tests** - Verify indexes improve query performance
5. **Full Cycle Tests** - Verify complete upgrade/downgrade cycle
6. **Timing Benchmarks** - Verify migrations complete within time limits

### Manual Testing

For complex migrations, manual testing is recommended:

```bash
# 1. Create test database
createdb reautomation_test

# 2. Apply migrations up to previous revision
alembic upgrade 002_voice_agent_integration

# 3. Insert test data
psql reautomation_test -c "INSERT INTO calls (phone_number, status, tier, created_at) VALUES ('+15555551234', 'completed', 'tier1', NOW());"

# 4. Apply new migration
alembic upgrade 003_context_management

# 5. Verify schema changes
psql reautomation_test -c "\d calls"

# 6. Verify data preserved
psql reautomation_test -c "SELECT * FROM calls;"

# 7. Test downgrade
alembic downgrade 002_voice_agent_integration

# 8. Verify rollback
psql reautomation_test -c "\d calls"
psql reautomation_test -c "SELECT * FROM calls;"
```

---

## Data Migrations

### When to Use Data Migrations

Create data migration scripts when:
- Adding new fields that need backfilling
- Transforming existing data
- Migrating data between tables
- Archiving old data

### Running Data Migrations

See `alembic/data_migrations/README.md` for detailed instructions.

**Quick reference:**
```bash
# After applying schema migration 003
python -m alembic.data_migrations.backfill_message_tokens
python -m alembic.data_migrations.backfill_context_fields
```

### Creating New Data Migrations

Follow the template in `alembic/data_migrations/backfill_context_fields.py`:

1. Create async class with `setup()`, `run()`, `teardown()` methods
2. Implement batched processing (100-500 records per batch)
3. Add progress logging
4. Make it idempotent (safe to re-run)
5. Document in `alembic/data_migrations/README.md`

---

## Troubleshooting

### Common Issues

**Issue: "Can't locate revision identified by..."**
```bash
# Solution: Check if migration file exists
ls alembic/versions/

# Reset to specific revision
alembic stamp 002_voice_agent_integration
```

**Issue: "Target database is not up to date"**
```bash
# Solution: Upgrade first
alembic upgrade head
```

**Issue: "Multiple head revisions present"**
```bash
# Solution: Merge branches
alembic merge heads -m "merge_branches"
```

**Issue: Migration fails mid-way**
```bash
# Solution 1: Check database state
alembic current

# Solution 2: Manually fix and stamp
# Fix database manually, then:
alembic stamp head

# Solution 3: Rollback and retry
alembic downgrade -1
alembic upgrade head
```

**Issue: Autogenerate doesn't detect changes**
- Ensure `target_metadata` in `alembic/env.py` imports your Base
- Check that models inherit from Base
- Some changes (indexes, constraints) may need manual migration

### Checking Migration Status

```bash
# Current revision
alembic current

# Pending migrations
alembic history

# Show SQL for pending migrations
alembic upgrade head --sql
```

### Database Recovery

**If migration corrupts database:**

1. **Restore from backup** (always backup before migrations!)
2. **Or rollback:**
```bash
alembic downgrade <previous_revision>
```
3. **Or fix manually and stamp:**
```sql
-- Manually fix database
-- Then update alembic version
UPDATE alembic_version SET version_num = '<correct_revision>';
```

---

## Production Deployment

### Pre-deployment Checklist

- [ ] All migrations tested locally (upgrade + downgrade)
- [ ] Migration tests pass (`pytest tests/migrations/`)
- [ ] Data migrations prepared (if needed)
- [ ] Database backup created
- [ ] Downtime window scheduled (if needed)
- [ ] Rollback plan documented
- [ ] Team notified of deployment

### Deployment Process

**1. Backup Database:**
```bash
pg_dump -h production-host -U postgres reautomation > backup_$(date +%Y%m%d_%H%M%S).sql
```

**2. Check Current State:**
```bash
# SSH to production
alembic current
alembic history
```

**3. Preview SQL:**
```bash
alembic upgrade head --sql > migration_preview.sql
# Review migration_preview.sql
```

**4. Apply Migration:**
```bash
# Set DATABASE_URL for production
export DATABASE_URL="postgresql://user:pass@prod-host:5432/reautomation"

# Apply migration
alembic upgrade head

# Verify
alembic current
```

**5. Run Data Migrations (if needed):**
```bash
python -m alembic.data_migrations.backfill_message_tokens
python -m alembic.data_migrations.backfill_context_fields
```

**6. Verify Application:**
```bash
# Start application
# Check logs for errors
# Run smoke tests
```

**7. Monitor:**
- Check application logs
- Monitor database performance
- Verify API endpoints working

### Rollback Procedure

**If migration fails:**

```bash
# 1. Stop application
systemctl stop reautomation

# 2. Rollback migration
alembic downgrade -1

# 3. Restore from backup if needed
psql reautomation < backup_20250106_140000.sql

# 4. Verify database state
alembic current

# 5. Restart application
systemctl start reautomation
```

### Zero-Downtime Migrations

For production systems that can't afford downtime:

**Phase 1: Add new columns (nullable)**
```python
def upgrade():
    op.add_column('calls', sa.Column('new_field', sa.String(), nullable=True))
```

**Phase 2: Deploy application that writes to both old and new columns**

**Phase 3: Backfill new column**
```bash
python -m alembic.data_migrations.backfill_new_field
```

**Phase 4: Make new column NOT NULL**
```python
def upgrade():
    op.alter_column('calls', 'new_field', nullable=False)
```

**Phase 5: Deploy application that only uses new column**

**Phase 6: Drop old column**
```python
def upgrade():
    op.drop_column('calls', 'old_field')
```

---

## Additional Resources

- [Alembic Documentation](https://alembic.sqlalchemy.org/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [PostgreSQL Index Documentation](https://www.postgresql.org/docs/current/indexes.html)

---

## Migration Checklist Template

Copy this for each new migration:

```markdown
## Migration: [NAME]

**Revision:** [REVISION_ID]
**Date:** [DATE]
**Author:** [NAME]

### Changes
- [ ] Schema changes documented
- [ ] Indexes added/modified
- [ ] Data migration needed? (Y/N)

### Testing
- [ ] Upgrade tested locally
- [ ] Downgrade tested locally
- [ ] Data preservation verified
- [ ] Index effectiveness verified
- [ ] Migration tests pass

### Deployment
- [ ] Backup created
- [ ] Downtime needed? (Y/N)
- [ ] Rollback plan documented
- [ ] Team notified

### Post-Deployment
- [ ] Migration applied successfully
- [ ] Data migrations run (if applicable)
- [ ] Application verified
- [ ] Monitoring checked
```
