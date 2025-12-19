# 🚀 Phase 2 Implementation Plan

**Feature**: OSM-Based Map Extraction (Phase 2)  
**Phase 2**: Visual Editor & YAML Serialization (Weeks 3–4)  
**Tasks**: T022-T035 (14 tasks total)  
**Status**: Starting now

---

## 📋 Phase 2 Task Breakdown

### Subphase 2a: YAML Schema & Serialization (T022-T025)

**Goal**: Define zones/routes in YAML with deterministic serialization

| Task | Function | Files | Priority | Est. Time |
|------|----------|-------|----------|-----------|
| T022 | `OSMZonesConfig` dataclass | osm_zones_yaml.py | P0 | 1h |
| T023 | `load_zones_yaml()` | osm_zones_yaml.py | P0 | 1h |
| T024 | `save_zones_yaml()` (deterministic) | osm_zones_yaml.py | P0 | 1.5h |
| T025 | `validate_zones_yaml()` | osm_zones_yaml.py | P0 | 1.5h |

**Deliverable**: Deterministic YAML serialization with validation

---

### Subphase 2b: Visual Editor Implementation (T026-T033)

**Goal**: Matplotlib-based GUI for zone/route editing

| Task | Feature | Files | Priority | Est. Time |
|------|---------|-------|----------|-----------|
| T026 | Editor class skeleton | osm_zones_editor.py | P0 | 1h |
| T027 | Click handlers (add vertices) | osm_zones_editor.py | P0 | 1.5h |
| T028 | Vertex editing (drag/delete) | osm_zones_editor.py | P0 | 1.5h |
| T029 | Undo/redo system | osm_zones_editor.py | P1 | 1h |
| T030 | Snapping logic | osm_zones_editor.py | P1 | 1h |
| T031 | Real-time validation | osm_zones_editor.py | P1 | 1h |
| T032 | Save to YAML | osm_zones_editor.py | P0 | 1h |
| T033 | Keyboard shortcuts & polish | osm_zones_editor.py | P1 | 1.5h |

**Deliverable**: Interactive visual editor with keyboard shortcuts

---

### Subphase 2c: Example & Integration (T034-T035)

| Task | Feature | Files | Priority | Est. Time |
|------|---------|-------|----------|-----------|
| T034 | Editor demo script | osm_map_editor_demo.py | P0 | 1h |
| T035 | Backward-compat smoke test | test_osm_backward_compat.py | P0 | 1.5h |

**Deliverable**: End-to-end demo + validation

---

## 🎯 Phase 2 Success Criteria

- [x] YAML schema defined and working
- [ ] Deterministic serialization (round-trip byte-identical)
- [ ] Visual editor functional and interactive
- [ ] All keyboard shortcuts working
- [ ] Backward-compatibility verified
- [ ] Full test coverage (unit + integration)
- [ ] Demo script ready

---

## 📦 Phase 2 Deliverables

### Files to Create
1. `robot_sf/maps/osm_zones_yaml.py` — YAML schema + serialization
2. `robot_sf/maps/osm_zones_editor.py` — Visual editor GUI
3. `examples/osm_map_editor_demo.py` — End-to-end demo
4. `tests/test_osm_zones_yaml.py` — YAML tests
5. `tests/test_osm_zones_editor.py` — Editor tests

### Key Technologies
- **PyYAML** (already in pyproject.toml)
- **Matplotlib** (already in pyproject.toml)
- **Shapely** (already available)

---

## 🔧 Implementation Order

1. **T022-T025**: YAML module first (foundation for everything)
2. **T026-T033**: Editor implementation (uses YAML for saving)
3. **T034-T035**: Examples & validation

---

## ✅ Starting Phase 2 Immediately

Ready to begin with T022: `OSMZonesConfig` dataclass definition.
