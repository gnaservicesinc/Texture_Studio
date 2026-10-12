# Loom Room workspace transition

Texture Studio 0.9.26, build 121.

The reported alert appeared because the category workspace still belonged to an indexed dataset while Loom Room now opened its original source folder. The existing workspace referenced `MatterLoom 2k`; all 1,824 registered original-map paths existed under `/opt/ipde/sources_mats`.

The backend now recognizes that one-way transition from a legacy reference dataset to its actual source folder. It checks existing ownership metadata and original paths without hashing or decoding the corpus, then rebinds the workspace under its lock. Existing categories, reviews, custom groups, classifier curation and model histories are retained. Conflicting or unrelated workspace ownership remains protected.

Loom Room remembers a workspace per canonical source root. Opening unrelated sources automatically chooses a separate reusable workspace instead of displaying another folder's categories or asking the user to resolve an ownership error.

Refresh also updates renamed references for automatically managed materials. Existing review status, notes, splits and size-review decisions are retained; new pending source proofs are verified when training needs them. Manually added records remain separate from automatic membership. This repairs the two Paper-category Cardboard references that still used the old folder name.

Category training inherits the active original source grid. A category's legacy 1024 default no longer silently changes a 2048 source-folder handoff. The explicitly requested crop projection contains exact 2048 samples; smaller originals remain available separately.

The live workspace was backed up in user storage before any migration.

The focused Release run passed 165 cases with zero failures and one opt-in UI-control skip in 22.856 seconds. Coverage includes migration with an approved saved model and custom category, unchanged reviews and samples, ownership-conflict rejection, renamed-reference preparation with deferred verification, unrelated-folder workspace selection/reuse, and a 2048 source-to-legacy-1024 category handoff. Dataset lifecycle, native dataset, preferences and training preparation regressions passed in the same run. `git diff --check` passed.

Installed refresh results follow below.
