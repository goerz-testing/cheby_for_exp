# Chebychev expansion of `exp(H dt) v`

* `cheby_for_exp.jl`: Minimum working example of the standard `exp(-1im * H * dt) v` adapted from https://github.com/JuliaQuantumControl/QuantumPropagators.jl/blob/master/src/cheby.jl
* `cheby_for_exp_relax.jl`: Adaptation for the "relax" propagation `exp(-1 * H  * dt) v` (which is commonly used to identify the ground state of `H`). Not working properly.

TODO: once the `relax` example is working, the final modification to `exp(H * dt) v` should be rather trivial (just changing the sign of `dt`)
