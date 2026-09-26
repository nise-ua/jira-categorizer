# Public Jira dataset

- Source: Apache Jira (https://issues.apache.org/jira)
- JQL: `project = KAFKA AND labels is not EMPTY AND component is not EMPTY ORDER BY created DESC`
- Count: 100
- Mapping: components -> impacted_area; labels -> labels
- Timestamps: created, updated (for surge detection)

## Top labels
- kip: 28
- needs-kip: 25
- gradle: 12
- flaky-test: 5
- streams: 5
- update: 4
- tiered-storage: 3
- build: 3
- queues-for-kafka: 3
- newbie: 3
- need-kip: 3
- refactoring: 2
- dependency: 2
- optimization: 2
- jacoco: 2

## Top impacted areas (components)
- streams: 50
- clients: 19
- consumer: 13
- core: 9
- tools: 8
- group-coordinator: 8
- build: 7
- producer : 6
- connect: 5
- tiered-storage: 2
- security: 2
- controller: 2
- documentation: 2
- metrics: 2
- system tests: 1
