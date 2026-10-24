# Ragnor licensing

## Current terms

Ragnor is source-available under **Elastic License 2.0 (ELv2)**
(SPDX `Elastic-2.0`). The unmodified [LICENSE](../LICENSE) controls;
[NOTICE](../NOTICE) preserves the project attribution. ELv2 is not an
OSI-approved open-source license. The package metadata and built artifacts
must identify the same license and include both files.

The maintainer requested this source change on 24 October 2026. It applies
to this revision and future distributions; previously published 0.4.1 and
earlier artifacts and tags retain their original licensing. The next release
carrying this revision must use a new version. No release is published by
this change.

## Commercial use and managed services

ELv2 permits use, modification and redistribution subject to its terms.
It restricts providing the software to third parties as a hosted or managed
service giving access to a substantial set of its features or functionality.
It also prohibits license-key circumvention and removal or obscuring of
licensing and copyright notices. Redistributions must include the terms;
modified copies must prominently state that they have been modified.
Commercial integration does not itself require payment.

These examples are illustrative, not additional license terms:

| Deployment | Typical interpretation, subject to the full terms |
| --- | --- |
| Internal enterprise agent | Generally permitted |
| Insurance or customer-service application using Ragnor privately | Generally permitted |
| Domain-specific customer Q&A | Generally permitted if substantial Ragnor functionality is not exposed |
| Managed ingestion/index/retrieval API exposing Ragnor to third parties | Potentially restricted; assess the actual exposed functionality |
| Redistributed self-hosted/on-premises bundle | Generally permitted with terms, notices and modification notices |

ELv2 does not prohibit every SaaS application. For borderline deployments,
consult the [official terms](https://www.elastic.co/licensing/elastic-license)
and [Elastic FAQ](https://www.elastic.co/licensing/elastic-license/faq), and
obtain advice appropriate to your deployment.

## Requests for separate rights

For a restricted managed-service offering, request a separately negotiated
license or clarification from the [repository maintainer](https://github.com/eslam-aboutaleb)
through the [repository issue tracker](https://github.com/eslam-aboutaleb/ragnor/issues).
Do not include confidential information in a public issue. This provides a
request route, not a commercial entitlement, price, warranty or SLA.
Additional rights require an explicit agreement with the rights holder.

## Provenance and release record

The source change is authorized by the maintainer's explicit task request.
The main-branch author inventory contains the maintainer's two name variants
with the same email. This inventory alone does not establish rights to any
copied or third-party material, and no legal review is claimed. Upstream
components retain their own licenses; this change does not relicense them.
Issue [#19](https://github.com/eslam-aboutaleb/ragnor/issues/19) remains open
for provenance/legal review, commercial-policy approval and release sign-off.

Release checks verify source consistency, exact license/notice contents in
wheel and sdist, license metadata, and clean installation. Do not rewrite
historical releases or publish a replacement 0.4.1 artifact under these terms.
