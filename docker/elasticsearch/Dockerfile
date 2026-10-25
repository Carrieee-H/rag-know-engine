# Elasticsearch with the IK Chinese analyzer.
# The IK plugin version must match the ES version exactly; change both together.
ARG ES_VERSION=8.15.3
FROM docker.elastic.co/elasticsearch/elasticsearch:${ES_VERSION}
ARG ES_VERSION
RUN bin/elasticsearch-plugin install --batch https://get.infini.cloud/elasticsearch/analysis-ik/${ES_VERSION}
# Baked in rather than bind-mounted: under OrbStack, opening a single-file bind mount from the
# macOS host blocked IK's Dictionary initialization (and with it shard recovery) indefinitely.
COPY IKAnalyzer.cfg.xml /usr/share/elasticsearch/config/analysis-ik/IKAnalyzer.cfg.xml
