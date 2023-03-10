# =========== #
# BUILD IMAGE #
# =========== #
FROM python:3.11-slim AS builder

ENV POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_NO_INTERACTION=1

ENV PATH="$POETRY_HOME/bin:$PATH"

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && curl -sSL https://install.python-poetry.org | python3 -

WORKDIR /app

# copy required files for the build
COPY poetry.lock pyproject.toml ./

# this will create the folder /app/.venv (might need adjustment depending on which poetry version you are using)
RUN poetry install --no-root --no-ansi --without dev

# ============= #
# RUNTIME IMAGE #
# ============= #
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# copy the venv folder from builder image 
COPY --from=builder /app/.venv ./.venv

# copy application
COPY json_steamroller ./

CMD [ "python", "executor.py" ]
