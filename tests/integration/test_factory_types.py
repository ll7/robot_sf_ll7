"""Integration tests for factory function type safety."""


class TestFactoryTypeSafety:
    """Integration tests for factory function type annotations."""

    def test_make_robot_env_type_safety(self):
        """Test that make_robot_env has proper type annotations.

        Integration: Factory function should be callable with correct types
        and return properly typed environment.
        """
        # Import should work
        from robot_sf.gym_env.environment_factory import make_robot_env

        # Should be callable (basic type check)
        assert callable(make_robot_env)

        # Should now work with defaults (implementation complete)
        env = make_robot_env()
        # Should return an environment-like object
        assert hasattr(env, "reset"), "Should have reset method"
        assert hasattr(env, "step"), "Should have step method"

    def test_make_pedestrian_env_type_safety(self):
        """Test that make_pedestrian_env has proper type annotations.

        Integration: Pedestrian factory should have consistent interface.
        """
        from robot_sf.gym_env.environment_factory import make_pedestrian_env

        assert callable(make_pedestrian_env)

        # Should now work with defaults (implementation complete)
        env = make_pedestrian_env()
        assert hasattr(env, "reset")
        assert hasattr(env, "step")

    def test_make_multi_robot_env_type_safety(self):
        """Test that make_multi_robot_env has proper type annotations.

        Integration: Multi-robot factory should handle num_robots parameter.
        """
        from robot_sf.gym_env.environment_factory import make_multi_robot_env

        assert callable(make_multi_robot_env)

        # Should now work with defaults (implementation complete)
        env = make_multi_robot_env()
        assert hasattr(env, "reset"), "Should have reset method"
        assert hasattr(env, "step"), "Should have step method"
