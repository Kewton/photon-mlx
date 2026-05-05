# PHOTON-RepoRAG Minimal Demo

## Product Overview

PHOTON-RepoRAG is a Multi-turn RAG system for local repositories and markdown corpora. It is designed for business Q&A where users ask follow-up questions, omit repeated context, compare related items, or add new conditions across turns.

## Business Problems

Conventional RAG often retrieves evidence from the current question only. This can fail when a user asks a non-standalone question such as "2号との違いは？" after asking about a specific制度 or policy in the previous turn.

PHOTON-RepoRAG keeps the current question, related past questions, and candidate evidence connected. The goal is to reduce missing evidence, unrelated context, and unstable follow-up answers.

## Baseline Mode

Baseline mode provides the normal repository or document RAG flow. It ingests files, builds an index, retrieves candidate chunks, and generates an answer with citations.

Baseline mode is useful for smoke testing because it does not require a PHOTON checkpoint. It still depends on the embedding and generation models configured in the selected YAML.

## PHOTON Mode

PHOTON mode adds a scoring layer before answer generation. It compares current-turn context, session-level context, and evidence candidates, then uses PHOTON score to help rank or prune evidence chunks.

This mode is intended for multi-turn scenarios where the current question alone does not contain enough information to select the right evidence.

## MVP Boundary

The v0.1.0 MVP is released through GitHub Release artifacts and source checkout. It does not publish checkpoints, external model weights, hosted demos, or PyPI packages.
