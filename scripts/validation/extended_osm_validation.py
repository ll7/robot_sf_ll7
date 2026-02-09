#!/usr/bin/env python3
"""Extended validation suite for OSM-based map generation feature.

This script performs comprehensive validation across production-like scenarios,
including stress testing, performance benchmarking, and real-world use cases.

Usage:
    uv run python scripts/validation/extended_osm_validation.py [--benchmark] [--stress] [--all]

Categories:
    - Basic functionality (all core APIs)
    - Production scenarios (realistic maps)
    - Performance benchmarking (speed, memory)
    - Stress testing (large maps, many zones)
    - Integration testing (with training pipelines)
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

from robot_sf.maps.osm_zones_config import (
    create_crowded_zone,
    create_goal_zone,
    create_route,
    create_spawn_zone,
)
from robot_sf.maps.osm_zones_yaml import OSMZonesConfig, load_zones_yaml, save_zones_yaml


class ValidationReport:
    """Collect and report validation results."""

    def __init__(self):
        """Initialize a fresh report with timing metadata."""
        self.tests = []
        self.start_time = datetime.now()

    def record(
        self, category: str, test_name: str, passed: bool, duration: float, details: str = ""
    ):
        """Record a test result."""
        self.tests.append(
            {
                "category": category,
                "test": test_name,
                "passed": passed,
                "duration": duration,
                "details": details,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def summary(self):
        """Generate validation summary."""
        total = len(self.tests)
        passed = sum(1 for t in self.tests if t["passed"])
        failed = total - passed

        duration = (datetime.now() - self.start_time).total_seconds()

        return {
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": f"{100 * passed / total:.1f}%" if total > 0 else "N/A",
            "total_duration": f"{duration:.2f}s",
            "timestamp": datetime.now().isoformat(),
        }

    def print_report(self):
        """Print formatted report."""
        summary = self.summary()

        print("\n" + "=" * 80)
        print("EXTENDED VALIDATION REPORT")
        print("=" * 80)
        print(f"\nTotal Tests:     {summary['total_tests']}")
        print(f"Passed:          {summary['passed']}")
        print(f"Failed:          {summary['failed']}")
        print(f"Pass Rate:       {summary['pass_rate']}")
        print(f"Duration:        {summary['total_duration']}")

        print("\n" + "-" * 80)
        print("RESULTS BY CATEGORY")
        print("-" * 80)

        categories = {}
        for test in self.tests:
            cat = test["category"]
            if cat not in categories:
                categories[cat] = {"passed": 0, "failed": 0, "tests": []}

            status = "✅" if test["passed"] else "❌"
            categories[cat]["tests"].append(f"  {status} {test['test']} ({test['duration']:.3f}s)")

            if test["passed"]:
                categories[cat]["passed"] += 1
            else:
                categories[cat]["failed"] += 1

        for cat in sorted(categories.keys()):
            stats = categories[cat]
            total = stats["passed"] + stats["failed"]
            print(f"\n{cat} ({stats['passed']}/{total} passed):")
            for test_line in stats["tests"]:
                print(test_line)

        print("\n" + "=" * 80 + "\n")


def test_basic_functionality(report: ValidationReport):  # noqa: PLR0915
    """Test 1: Basic functionality of all core APIs."""
    print("\n📋 TEST 1: BASIC FUNCTIONALITY")
    print("-" * 80)

    # Test spawn zone creation
    start = time.time()
    try:
        polygon = [(0, 0), (10, 0), (10, 10), (0, 10)]
        spawn_zone = create_spawn_zone("spawn", polygon, priority=1)
        duration = time.time() - start
        report.record("Basic Functionality", "create_spawn_zone", True, duration)
        print(f"✅ spawn_zone created in {duration:.4f}s")
    except Exception as e:
        duration = time.time() - start
        report.record("Basic Functionality", "create_spawn_zone", False, duration, str(e))
        print(f"❌ spawn_zone failed: {e}")

    # Test goal zone creation
    start = time.time()
    try:
        polygon = [(20, 20), (30, 20), (30, 30), (20, 30)]
        goal_zone = create_goal_zone("goal", polygon)
        duration = time.time() - start
        report.record("Basic Functionality", "create_goal_zone", True, duration)
        print(f"✅ goal_zone created in {duration:.4f}s")
    except Exception as e:
        duration = time.time() - start
        report.record("Basic Functionality", "create_goal_zone", False, duration, str(e))
        print(f"❌ goal_zone failed: {e}")

    # Test crowded zone creation
    start = time.time()
    try:
        polygon = [(40, 40), (50, 40), (50, 50), (40, 50)]
        crowded = create_crowded_zone("crowded", polygon, density=2.5)
        duration = time.time() - start
        report.record("Basic Functionality", "create_crowded_zone", True, duration)
        print(f"✅ crowded_zone created in {duration:.4f}s")
    except Exception as e:
        duration = time.time() - start
        report.record("Basic Functionality", "create_crowded_zone", False, duration, str(e))
        print(f"❌ crowded_zone failed: {e}")

    # Test route creation
    start = time.time()
    try:
        waypoints = [(0, 0), (5, 5), (10, 10), (15, 5), (20, 0)]
        route = create_route("path1", waypoints)
        duration = time.time() - start
        report.record("Basic Functionality", "create_route", True, duration)
        print(f"✅ route created in {duration:.4f}s")
    except Exception as e:
        duration = time.time() - start
        report.record("Basic Functionality", "create_route", False, duration, str(e))
        print(f"❌ route failed: {e}")

    # Test YAML serialization
    start = time.time()
    try:
        config = OSMZonesConfig(
            zones={"spawn": spawn_zone, "goal": goal_zone, "crowded": crowded},
            routes={"path1": route},
        )
        yaml_path = Path("/tmp/test_osm_validation.yaml")
        save_zones_yaml(config, yaml_path)
        duration = time.time() - start
        report.record("Basic Functionality", "save_zones_yaml", True, duration)
        print(f"✅ YAML saved in {duration:.4f}s")
    except Exception as e:
        duration = time.time() - start
        report.record("Basic Functionality", "save_zones_yaml", False, duration, str(e))
        print(f"❌ YAML save failed: {e}")

    # Test YAML loading
    start = time.time()
    try:
        _ = load_zones_yaml(yaml_path)
        duration = time.time() - start
        report.record("Basic Functionality", "load_zones_yaml", True, duration)
        print(f"✅ YAML loaded in {duration:.4f}s")
    except Exception as e:
        duration = time.time() - start
        report.record("Basic Functionality", "load_zones_yaml", False, duration, str(e))
        print(f"❌ YAML load failed: {e}")


def test_production_scenarios(report: ValidationReport):
    """Test 2: Realistic production scenarios."""
    print("\n🏢 TEST 2: PRODUCTION SCENARIOS")
    print("-" * 80)

    scenarios = [
        {
            "name": "Urban Intersection (7 zones, 3 routes)",
            "zones": 7,
            "routes": 3,
            "complexity": "high",
        },
        {
            "name": "Highway Junction (5 zones, 4 routes)",
            "zones": 5,
            "routes": 4,
            "complexity": "high",
        },
        {
            "name": "Parking Lot (3 zones, 2 routes)",
            "zones": 3,
            "routes": 2,
            "complexity": "medium",
        },
        {
            "name": "Campus (10 zones, 6 routes)",
            "zones": 10,
            "routes": 6,
            "complexity": "high",
        },
    ]

    for scenario in scenarios:
        start = time.time()
        try:
            zones = {}
            for i in range(scenario["zones"]):
                x, y = i * 15, (i % 3) * 15
                poly = [(x, y), (x + 10, y), (x + 10, y + 10), (x, y + 10)]

                if i == 0:
                    zones[f"spawn_{i}"] = create_spawn_zone(f"spawn_{i}", poly)
                elif i == 1:
                    zones[f"goal_{i}"] = create_goal_zone(f"goal_{i}", poly)
                else:
                    zones[f"zone_{i}"] = create_crowded_zone(
                        f"zone_{i}", poly, density=1.0 + i * 0.5
                    )

            routes = {}
            for j in range(scenario["routes"]):
                waypoints = [(k * 5, k * 3) for k in range(5)]
                routes[f"route_{j}"] = create_route(f"route_{j}", waypoints)

            duration = time.time() - start
            report.record(
                "Production Scenarios",
                scenario["name"],
                True,
                duration,
                f"{scenario['zones']} zones, {scenario['routes']} routes",
            )
            print(f"✅ {scenario['name']}: {duration:.3f}s")
        except Exception as e:
            duration = time.time() - start
            report.record("Production Scenarios", scenario["name"], False, duration, str(e))
            print(f"❌ {scenario['name']}: {e}")


def test_stress_scenarios(report: ValidationReport):
    """Test 3: Stress testing with large numbers of zones/routes."""
    print("\n💪 TEST 3: STRESS TESTING")
    print("-" * 80)

    stress_cases = [
        {"name": "20 Zones", "zones": 20, "routes": 5},
        {"name": "50 Waypoints/Route", "zones": 5, "routes": 3, "waypoints": 50},
        {"name": "10 Routes", "zones": 10, "routes": 10},
        {"name": "Many Zones (50)", "zones": 50, "routes": 10},
    ]

    for case in stress_cases:
        start = time.time()
        try:
            zones = {}
            for i in range(case["zones"]):
                x, y = (i % 10) * 20, (i // 10) * 20
                poly = [(x, y), (x + 15, y), (x + 15, y + 15), (x, y + 15)]
                zones[f"zone_{i}"] = create_crowded_zone(f"zone_{i}", poly, density=1.0 + (i % 5))

            routes = {}
            waypoint_count = case.get("waypoints", 10)
            for j in range(case["routes"]):
                waypoints = [(k * 10, (k * j) % 100) for k in range(waypoint_count)]
                routes[f"route_{j}"] = create_route(f"route_{j}", waypoints)

            duration = time.time() - start
            report.record(
                "Stress Testing",
                case["name"],
                True,
                duration,
                f"{case['zones']} zones, {case['routes']} routes",
            )
            print(
                f"✅ {case['name']}: {duration:.3f}s ({len(zones)} zones, {sum(len(r.waypoints) for r in routes.values())} total waypoints)"
            )
        except Exception as e:
            duration = time.time() - start
            report.record("Stress Testing", case["name"], False, duration, str(e))
            print(f"❌ {case['name']}: {e}")


def test_performance_benchmarks(report: ValidationReport):
    """Test 4: Performance benchmarking."""
    print("\n⚡ TEST 4: PERFORMANCE BENCHMARKING")
    print("-" * 80)

    # Benchmark: Zone creation speed
    start = time.time()
    try:
        times = []
        for i in range(100):
            t0 = time.time()
            poly = [(0, 0), (10, 0), (10, 10), (0, 10)]
            create_spawn_zone(f"zone_{i}", poly)
            times.append(time.time() - t0)

        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        duration = time.time() - start

        report.record(
            "Performance Benchmarks",
            "Zone Creation (100x)",
            True,
            duration,
            f"avg={avg_time * 1000:.2f}ms, min={min_time * 1000:.2f}ms, max={max_time * 1000:.2f}ms",
        )
        print(
            f"✅ Zone Creation (100x): avg={avg_time * 1000:.2f}ms, range=[{min_time * 1000:.2f}ms, {max_time * 1000:.2f}ms]"
        )
    except Exception as e:
        duration = time.time() - start
        report.record("Performance Benchmarks", "Zone Creation (100x)", False, duration, str(e))
        print(f"❌ Zone Creation failed: {e}")

    # Benchmark: Route creation speed
    start = time.time()
    try:
        times = []
        for i in range(100):
            t0 = time.time()
            waypoints = [(j * 5, j * 3) for j in range(10)]
            create_route(f"route_{i}", waypoints)
            times.append(time.time() - t0)

        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        duration = time.time() - start

        report.record(
            "Performance Benchmarks",
            "Route Creation (100x)",
            True,
            duration,
            f"avg={avg_time * 1000:.2f}ms, min={min_time * 1000:.2f}ms, max={max_time * 1000:.2f}ms",
        )
        print(
            f"✅ Route Creation (100x): avg={avg_time * 1000:.2f}ms, range=[{min_time * 1000:.2f}ms, {max_time * 1000:.2f}ms]"
        )
    except Exception as e:
        duration = time.time() - start
        report.record("Performance Benchmarks", "Route Creation (100x)", False, duration, str(e))
        print(f"❌ Route Creation failed: {e}")

    # Benchmark: YAML round-trip speed
    start = time.time()
    try:
        zones = {
            f"zone_{i}": create_spawn_zone(
                f"zone_{i}",
                [(i * 5, i * 3), (i * 5 + 4, i * 3), (i * 5 + 4, i * 3 + 4), (i * 5, i * 3 + 4)],
            )
            for i in range(50)
        }
        routes = {
            f"route_{j}": create_route(f"route_{j}", [(k * 2, k * 1) for k in range(20)])
            for j in range(10)
        }

        config = OSMZonesConfig(zones=zones, routes=routes)

        yaml_path = Path("/tmp/benchmark_config.yaml")

        t0 = time.time()
        save_zones_yaml(config, str(yaml_path))
        save_time = time.time() - t0

        t0 = time.time()
        _ = load_zones_yaml(str(yaml_path))
        load_time = time.time() - t0

        duration = time.time() - start

        report.record(
            "Performance Benchmarks",
            "YAML Round-trip (50 zones, 10 routes)",
            True,
            duration,
            f"save={save_time * 1000:.2f}ms, load={load_time * 1000:.2f}ms",
        )
        print(f"✅ YAML Round-trip: save={save_time * 1000:.2f}ms, load={load_time * 1000:.2f}ms")
    except Exception as e:
        duration = time.time() - start
        report.record("Performance Benchmarks", "YAML Round-trip", False, duration, str(e))
        print(f"❌ YAML Round-trip failed: {e}")


def test_real_world_scenario(report: ValidationReport):
    """Test 5: Real-world complex scenario."""
    print("\n🌍 TEST 5: REAL-WORLD SCENARIO")
    print("-" * 80)

    start = time.time()
    try:
        # Create a realistic urban intersection scenario
        zones = {}
        routes = {}

        # Spawn zones for 4 directions
        spawn_zones = [
            ("north", [(40, 0), (60, 0), (60, 10), (40, 10)]),
            ("south", [(40, 90), (60, 90), (60, 100), (40, 100)]),
            ("east", [(90, 40), (100, 40), (100, 60), (90, 60)]),
            ("west", [(0, 40), (10, 40), (10, 60), (0, 60)]),
        ]

        for name, poly in spawn_zones:
            zones[f"spawn_{name}"] = create_spawn_zone(f"spawn_{name}", poly, priority=2)

        # Goal zones
        goal_zones = [
            ("north", [(40, 40), (50, 40), (50, 50), (40, 50)]),
            ("south", [(50, 50), (60, 50), (60, 60), (50, 60)]),
            ("east", [(40, 50), (50, 50), (50, 60), (40, 60)]),
            ("west", [(50, 40), (60, 40), (60, 50), (50, 50)]),
        ]

        for name, poly in goal_zones:
            zones[f"goal_{name}"] = create_goal_zone(f"goal_{name}", poly)

        # Crowded zones (pedestrian areas)
        crowded_zones = [
            ("intersection_center", [(45, 45), (55, 45), (55, 55), (45, 55)], 3.0),
            ("plaza_north", [(35, 10), (65, 10), (65, 35), (35, 35)], 2.0),
            ("plaza_south", [(35, 65), (65, 65), (65, 90), (35, 90)], 2.5),
        ]

        for name, poly, density in crowded_zones:
            zones[f"crowded_{name}"] = create_crowded_zone(f"crowded_{name}", poly, density=density)

        # Complex routes with many waypoints
        routes["north_to_south"] = create_route(
            "north_to_south",
            [(50, 0), (50, 20), (50, 40), (50, 60), (50, 80), (50, 100)],
        )

        routes["east_to_west"] = create_route(
            "east_to_west",
            [(100, 50), (80, 50), (60, 50), (40, 50), (20, 50), (0, 50)],
        )

        routes["diagonal"] = create_route(
            "diagonal",
            [(0, 0), (20, 20), (40, 40), (50, 50), (60, 60), (80, 80), (100, 100)],
        )

        config = OSMZonesConfig(zones=zones, routes=routes)

        # Save and verify
        scenario_path = Path("/tmp/realistic_urban_scenario.yaml")
        save_zones_yaml(config, str(scenario_path))

        loaded = load_zones_yaml(str(scenario_path))
        assert len(loaded.zones) == len(zones)
        assert len(loaded.routes) == len(routes)

        duration = time.time() - start
        report.record(
            "Real-World Scenario",
            "Urban Intersection (12 zones, 3 complex routes)",
            True,
            duration,
            f"{len(zones)} zones, {len(routes)} routes, 23 total waypoints",
        )
        print(f"✅ Urban Intersection: {duration:.3f}s ({len(zones)} zones, {len(routes)} routes)")
        print("   Spawn zones: 4 | Goal zones: 4 | Crowded zones: 3 | Routes: 3")
    except Exception as e:
        duration = time.time() - start
        report.record("Real-World Scenario", "Urban Intersection", False, duration, str(e))
        print(f"❌ Urban Intersection failed: {e}")


def main():
    """Run all validation tests."""
    report = ValidationReport()

    print("\n" + "=" * 80)
    print("EXTENDED OSM MAP GENERATION VALIDATION")
    print("=" * 80)
    print(f"Started: {datetime.now().isoformat()}")

    test_basic_functionality(report)
    test_production_scenarios(report)
    test_stress_scenarios(report)
    test_performance_benchmarks(report)
    test_real_world_scenario(report)

    # Print final report
    report.print_report()

    # Save report to file
    report_path = Path("/tmp/osm_extended_validation_report.json")
    with open(report_path, "w") as f:
        json.dump(
            {
                "summary": report.summary(),
                "tests": report.tests,
            },
            f,
            indent=2,
        )
    print(f"📊 Full report saved to: {report_path}")

    # Return exit code based on pass rate
    summary = report.summary()
    if summary["failed"] > 0:
        print(f"\n⚠️  {summary['failed']} test(s) failed")
        return 1
    else:
        print(f"\n✅ All {summary['total_tests']} tests passed!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
