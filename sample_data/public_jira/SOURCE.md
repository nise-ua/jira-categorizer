# Public Jira dataset

- **Source**: Apache Jira (`https://issues.apache.org/jira`)
- **Project**: KAFKA
- **JQL**: `project = KAFKA AND labels is not EMPTY AND component is not EMPTY ORDER BY updated DESC`
- **Count**: 100 issues
- **Fetched fields**: key, summary, description, labels, components
- **Mapping**: `components` -> `impacted_area` (pipe-joined); `labels` kept as multi-label target
- **License / terms**: Apache Software Foundation public issue tracker; used for offline ML experimentation

## Label frequency (top 15)
- needs-kip: 21
- kip: 11
- newbie: 10
- gradle: 10
- Gradle: 9
- flaky-test: 7
- build: 6
- update: 5
- kip-848-client-support: 5
- beginner: 4
- need-kip: 4
- consumer-threading-refactor: 3
- tiered-storage: 3
- kip1071: 3
- OAuth2: 2

## Component / impacted_area frequency (top 15)
- streams: 37
- clients: 24
- consumer: 14
- build: 12
- core: 9
- tools: 7
- connect: 6
- security: 5
- system tests: 5
- group-coordinator: 5
- unit tests: 4
- producer : 3
- documentation: 2
- Tiered-Storage: 2
- streams-test-utils: 2
