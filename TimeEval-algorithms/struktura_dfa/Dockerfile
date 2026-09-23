FROM registry.gitlab.hpi.de/akita/i/rust-base

LABEL maintainer="webfastonline@gmail.com"

# Copy source and build in release mode
COPY Cargo.toml ./
COPY src/ ./src/
RUN cargo build --release && \
    cp target/release/struktura-timeeval /app/ && \
    rm -rf target/

ENV ALGORITHM_MAIN="/app/struktura-timeeval"
