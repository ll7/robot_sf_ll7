
# Benchmarks and Optimizations

## Zero-Load Simulator Benchmark
This benchmark measures the overall simulation performance
without any training overhead for fitting models, etc.
It can be seen as a lower bound for evaluating the training time.

### Profiler Results (2022-10-13)
Following profiler results were yielded by git hash 64f81ecb6fc7e12059ebfda46b31251759f47cf5:

```text
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
3308264/2709244  589.983    0.000  997.865    0.000 {built-in method numpy.core._multiarray_umath.implement_array_function}
   210577  183.302    0.001  183.302    0.001 /home/marco-pc/.local/lib/python3.8/site-packages/numpy/core/shape_base.py:432(<listcomp>)
    40004  120.771    0.003 1192.513    0.030 /home/marco-pc/source-code/robot-sf/robot_sf/extenders_py_sf/extender_scene.py:41(get_states)
    40004   98.123    0.002   98.123    0.002 /home/marco-pc/source-code/robot-sf/robot_sf/extenders_py_sf/extender_scene.py:42(<listcomp>)
   210577   89.084    0.000   89.084    0.000 /home/marco-pc/.local/lib/python3.8/site-packages/numpy/core/shape_base.py:424(<setcomp>)
   210577   72.057    0.000  113.210    0.001 /home/marco-pc/.local/lib/python3.8/site-packages/numpy/core/shape_base.py:420(<listcomp>)
   375958   45.013    0.000   45.312    0.000 /home/marco-pc/.local/lib/python3.8/site-packages/pysocialforce/utils/stateutils.py:46(normalize)
937310459   41.244    0.000   41.244    0.000 {built-in method numpy.asanyarray}
    20151   20.868    0.001   20.898    0.001 /home/marco-pc/source-code/robot-sf/robot_sf/range_sensor.py:130(simple_raycast)
    41847    8.586    0.000    8.587    0.000 {built-in method builtins.max}
  2119529    7.892    0.000    7.892    0.000 {method 'reduce' of 'numpy.ufunc' objects}
    20000    7.214    0.000   54.796    0.003 /home/marco-pc/.local/lib/python3.8/site-packages/pysocialforce/forces.py:303(_get_force)
   121994    7.052    0.000    7.054    0.000 {method 'sub' of 're.Pattern' objects}
    20000    6.916    0.000    9.776    0.000 /home/marco-pc/.local/lib/python3.8/site-packages/pysocialforce/forces.py:116(_get_force)
    20002    5.030    0.000  610.048    0.030 /home/marco-pc/source-code/robot-sf/robot_sf/extenders_py_sf/extender_sim.py:161(get_pedestrians_positions)
    20000    4.644    0.000   15.689    0.001 /home/marco-pc/source-code/robot-sf/robot_sf/extenders_py_sf/extender_force.py:131(_get_force)
   266683    4.208    0.000    6.039    0.000 /home/marco-pc/source-code/robot-sf/robot_sf/map.py:247(n_closest_fill)
    60827    4.167    0.000   19.894    0.000 /home/marco-pc/source-code/robot-sf/robot_sf/map.py:245(fill_surrounding)
    49915    3.537    0.000    9.118    0.000 /home/marco-pc/source-code/robot-sf/robot_sf/utils/utilities.py:31(lines_intersection)
    36521    3.366    0.000    3.910    0.000 /home/marco-pc/.local/lib/python3.8/site-packages/llvmlite/binding/ffi.py:149(__call__)
   321768    2.818    0.000    5.442    0.000 /home/marco-pc/.local/lib/python3.8/site-packages/pysocialforce/utils/stateutils.py:81(each_diff)
    20151    2.313    0.000    2.327    0.000 /home/marco-pc/source-code/robot-sf/robot_sf/range_sensor.py:157(range_postprocessing)
    20000    2.310    0.000   16.872    0.001 /home/marco-pc/source-code/robot-sf/robot_sf/utils/utilities.py:73(change_direction)
   432603    2.236    0.000    4.247    0.000 /home/marco-pc/.local/lib/python3.8/site-packages/numpy/linalg/linalg.py:2357(norm)
   463294    2.150    0.000    2.150    0.000 {method 'copy' of 'numpy.ndarray' objects}
   439320    1.977    0.000    4.144    0.000 /home/marco-pc/.local/lib/python3.8/site-packages/numpy/lib/shape_base.py:1191(tile)
    20000    1.928    0.000    5.603    0.000 /home/marco-pc/.local/lib/python3.8/site-packages/pysocialforce/forces.py:191(_get_force)
    80978    1.891    0.000    2.734    0.000 /home/marco-pc/source-code/robot-sf/robot_sf/map.py:195(convert_world_to_grid_no_error)
    20000    1.819    0.000   23.570    0.001 /home/marco-pc/source-code/robot-sf/robot_sf/extenders_py_sf/extender_force.py:85(_get_force)
   301768    1.684    0.000    5.217    0.000 /home/marco-pc/source-code/robot-sf/robot_sf/extenders_py_sf/extender_force.py:23(normalize)
    57383    1.520    0.000    2.222    0.000 /home/marco-pc/.local/lib/python3.8/site-packages/pysocialforce/forces.py:205(<listcomp>)
  1806048    1.516    0.000    1.516    0.000 {method 'reshape' of 'numpy.ndarray' objects}
    20002    1.498    0.000 1200.477    0.060 /home/marco-pc/source-code/robot-sf/robot_sf/extenders_py_sf/extender_sim.py:172(active_peds_update)
    20000    1.423    0.000    6.134    0.000 /home/marco-pc/.local/lib/python3.8/site-packages/pysocialforce/forces.py:262(_get_force)
   210577    1.269    0.000  868.045    0.004 /home/marco-pc/.local/lib/python3.8/site-packages/numpy/core/shape_base.py:357(stack)
    20000    1.222    0.000 1214.864    0.061 /home/marco-pc/source-code/robot-sf/robot_sf/extenders_py_sf/extender_sim.py:203(update_peds_on_scene)
   121658    1.180    0.000    1.487    0.000 /home/marco-pc/.local/lib/python3.8/site-packages/numpy/lib/stride_tricks.py:340(_broadcast_to)
  1201095    1.141    0.000    1.141    0.000 {built-in method numpy.array}
   459508    1.037    0.000    1.037    0.000 {method 'repeat' of 'numpy.ndarray' objects}
   624678    1.018    0.000    3.700    0.000 /home/marco-pc/.local/lib/python3.8/site-packages/numpy/core/fromnumeric.py:69(_wrapreduction)
    20002    1.002    0.000  588.848    0.029 /home/marco-pc/source-code/robot-sf/robot_sf/extenders_py_sf/extender_sim.py:167(get_pedestrians_groups)
   466228    0.960    0.000    1.173    0.000 /home/marco-pc/.local/lib/python3.8/site-packages/pysocialforce/utils/stateutils.py:106(center_of_mass)
  2042550    0.947    0.000    0.947    0.000 /home/marco-pc/.local/lib/python3.8/site-packages/numba/core/serialize.py:29(_numba_unpickle)
    60177    0.941    0.000    1.664    0.000 /home/marco-pc/source-code/robot-sf/robot_sf/map.py:157(check_if_valid_world_coordinates)
   549113    0.889    0.000    0.889    0.000 {built-in method numpy.zeros}
    59830    0.858    0.000    0.858    0.000 /home/marco-pc/source-code/robot-sf/robot_sf/utils/utilities.py:60(rotate_segment)
867508/60713    0.834    0.000    1.834    0.000 /usr/lib/python3.8/copy.py:128(deepcopy)
   321768    0.832    0.000    1.195    0.000 /home/marco-pc/.local/lib/python3.8/site-packages/numpy/lib/twodim_base.py:162(eye)
   321768    0.688    0.000    0.886    0.000 /home/marco-pc/.local/lib/python3.8/site-packages/pysocialforce/utils/stateutils.py:72(vec_diff)
   680919    0.688    0.000    0.827    0.000 /home/marco-pc/.local/lib/python3.8/site-packages/pysocialforce/scene.py:52(pos)
```

**Interpretation:**
It becomes clear that most computation effort is related to NumPy array processing (60-70%).
The other significant efforts are related to the PySocialForce simulator extension (10%).
All remaining parts of the codebase are not contributing much in particular.

As NumPy function calls are all over the place this means that each portion of the codebase
needs to be profiled separately to determine the hot spots. Most likely, the map implementation
and the simulator extension are good candidates for an optimization.
