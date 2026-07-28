from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import splu

from fno_tps.config import StudyConfig
from fno_tps.materials import EvalMode, RegionProperties, build_region_properties
from fno_tps.problem import SimulationCase


Array = np.ndarray
SourceFunction = Callable[[Array, Array, float], Array]
FluxIntegralFunction = Callable[[Array, float, float], Array]


@dataclass(frozen=True)
class BlockGrid:
    x_faces: Array
    x_centers: Array
    dx: Array
    y_faces: Array
    y_centers: Array
    dy: float
    region: Array
    nx_tps: int
    nx_bond: int
    nx_back: int

    @property
    def nx(self) -> int:
        return len(self.x_centers)

    @property
    def ny(self) -> int:
        return len(self.y_centers)

    @property
    def bond_slice(self) -> slice:
        return slice(self.nx_tps, self.nx_tps + self.nx_bond)

    @property
    def backing_slice(self) -> slice:
        return slice(self.nx_tps + self.nx_bond, self.nx)

    @property
    def structural_face_index(self) -> int:
        return self.nx_tps + self.nx_bond - 1


@dataclass
class Trajectory:
    times: Array
    x_centers: Array
    y_centers: Array
    dx: Array
    temperatures: Array
    surface_temperatures: Array
    energy_times: Array
    internal_energy: Array
    expected_energy: Array
    energy_residual: Array
    boundary_input_energy: Array
    radiated_energy: Array
    net_boundary_energy: Array
    nonlinear_iteration_counts: Array
    minimum_temperature: float
    maximum_hot_face_temperature: float
    factorization_count: int
    step_driver: str = "linear"
    property_model: str = "constant_effective"
    linear_solve_count: int = 0
    max_nonlinear_iterations: int = 0
    iteration_range_clamps: int = 0
    accepted_range_excursions: int = 0
    solver_converged: bool = True
    nonlinear_final_residual_norms: Array = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    nonlinear_damped_iteration_counts: Array = field(
        default_factory=lambda: np.empty(0, dtype=np.int32)
    )
    nonlinear_backtrack_counts: Array = field(
        default_factory=lambda: np.empty(0, dtype=np.int32)
    )
    failed_nonlinear_iterations: int = 0
    property_query_temperature_range: tuple[float, float] = (
        float("nan"),
        float("nan"),
    )
    boundary_energy_increment: Array = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float64)
    )
    observed_temperature_range: tuple[float, float] = (float("nan"), float("nan"))

    @property
    def delta_temperature(self) -> Array:
        return self.temperatures - self.temperatures[0:1]

    @property
    def final_nonlinear_residual_norm(self) -> float:
        if self.nonlinear_final_residual_norms.size == 0:
            return 0.0
        return float(self.nonlinear_final_residual_norms[-1])


def build_block_grid(
    config: StudyConfig,
    d_tps: float,
    *,
    bond_thickness: float | None = None,
    nx_bond: int | None = None,
) -> BlockGrid:
    d_bond = config.bond_thickness if bond_thickness is None else float(bond_thickness)
    n_bond = config.mesh.nx_bond if nx_bond is None else int(nx_bond)
    if d_bond < 0.0 or n_bond < 0 or (d_bond == 0.0) != (n_bond == 0):
        raise ValueError("bond_thickness and nx_bond must either both be zero or both positive.")
    if d_tps <= 0.0:
        raise ValueError("d_tps must be positive.")

    tps_faces = np.linspace(0.0, d_tps, config.mesh.nx_tps + 1)
    face_parts = [tps_faces]
    region_parts = [np.zeros(config.mesh.nx_tps, dtype=np.int8)]
    x_cursor = d_tps
    if n_bond:
        bond_faces = np.linspace(x_cursor, x_cursor + d_bond, n_bond + 1)
        face_parts.append(bond_faces[1:])
        region_parts.append(np.ones(n_bond, dtype=np.int8))
        x_cursor += d_bond
    back_faces = np.linspace(
        x_cursor,
        x_cursor + config.backing_thickness,
        config.mesh.nx_back + 1,
    )
    face_parts.append(back_faces[1:])
    region_parts.append(np.full(config.mesh.nx_back, 2, dtype=np.int8))
    x_faces = np.concatenate(face_parts)
    dx = np.diff(x_faces)
    x_centers = 0.5 * (x_faces[:-1] + x_faces[1:])
    y_faces = np.linspace(0.0, config.lateral_length, config.mesh.ny + 1)
    y_centers = 0.5 * (y_faces[:-1] + y_faces[1:])
    return BlockGrid(
        x_faces=x_faces,
        x_centers=x_centers,
        dx=dx,
        y_faces=y_faces,
        y_centers=y_centers,
        dy=float(y_faces[1] - y_faces[0]),
        region=np.concatenate(region_parts),
        nx_tps=config.mesh.nx_tps,
        nx_bond=n_bond,
        nx_back=config.mesh.nx_back,
    )


class TPSFVSolver:
    """Conservative cell-centered block-mesh Crank–Nicolson solver.

    ``k_x``, ``k_y``, ``rho_c``, ``g_x``, ``g_y``, and ``conduction`` are
    live attributes. ``k`` remains an alias for ``k_x`` for compatibility.
    On the nonlinear path they are updated to the property state used by the
    most recent assembly.
    """

    def __init__(
        self,
        config: StudyConfig,
        case: SimulationCase,
        *,
        bond_thickness: float | None = None,
        nx_bond: int | None = None,
        bond_conductivity_override: Array | None = None,
        bond_rho_c_override: float | None = None,
        interface_resistance_y: Array | None = None,
        source_fn: SourceFunction | None = None,
        flux_integral_fn: FluxIntegralFunction | None = None,
        region_properties: RegionProperties | None = None,
        rear_temperature: float | None = None,
    ):
        self.config = config
        self.case = case
        self.grid = build_block_grid(
            config,
            case.d_tps,
            bond_thickness=bond_thickness,
            nx_bond=nx_bond,
        )
        self.dt = config.time.dt
        self.source_fn = source_fn
        self.flux_integral_fn = flux_integral_fn
        self.rear_temperature = (
            None if rear_temperature is None else float(rear_temperature)
        )
        self.interface_resistance_y = (
            None
            if interface_resistance_y is None
            else np.asarray(interface_resistance_y, dtype=np.float64)
        )
        if self.interface_resistance_y is not None:
            if self.grid.nx_bond != 0:
                raise ValueError("Face resistance mode requires a zero-thickness bond grid.")
            if self.interface_resistance_y.shape != (self.grid.ny,):
                raise ValueError("interface_resistance_y must have shape (Ny,).")
            if np.any(self.interface_resistance_y <= 0.0):
                raise ValueError("interface resistance must be positive.")

        bond_k = (
            case.bond_conductivity(self.grid.y_centers, config)
            if bond_conductivity_override is None
            else np.asarray(bond_conductivity_override, dtype=np.float64)
        )
        if bond_k.shape != (self.grid.ny,) or np.any(bond_k <= 0.0):
            raise ValueError("Bond conductivity must be a positive (Ny,) array.")
        self.bond_conductivity = bond_k
        self.bond_rho_c = (
            config.bond.rho_c if bond_rho_c_override is None else float(bond_rho_c_override)
        )
        if self.bond_rho_c <= 0.0:
            raise ValueError("Bond volumetric heat capacity must be positive.")

        self.region_properties: RegionProperties = (
            build_region_properties(config)
            if region_properties is None
            else region_properties
        )
        self._use_linear_step = (
            not self.region_properties.is_temperature_dependent
            and (
                not config.surface.reradiation_enabled
                or config.surface.radiation_coupling == "boundary_response"
            )
            and self.rear_temperature is None
        )
        initial_field = np.full(
            (self.grid.nx, self.grid.ny),
            config.initial_temperature,
            dtype=np.float64,
        )
        initial_k_x, initial_k_y, initial_rho_c = (
            self.directional_properties_from_temperature(
                initial_field,
                mode="accepted",
            )
        )
        self.k_x = initial_k_x.copy()
        self.k_y = initial_k_y.copy()
        self.k = self.k_x
        self.rho_c = initial_rho_c.copy()

        self.volume = self.grid.dx[:, None] * self.grid.dy
        self.mass = self.rho_c * self.volume
        self.g_x = self._x_conductances()
        self.g_y = self._y_conductances()
        self.conduction = self._assemble_conduction()
        self._apply_rear_dirichlet()
        self._initialize_sparse_pattern()
        mass_diagonal = sparse.diags(self.mass.reshape(-1))
        self.matrix_a = (mass_diagonal - 0.5 * self.dt * self.conduction).tocsc()
        self.matrix_b = (mass_diagonal + 0.5 * self.dt * self.conduction).tocsr()
        self.factor = splu(self.matrix_a)
        self.factorization_count = 1
        self.linear_solve_count = 0
        self.surface_heat_transfer_coefficient = (
            2.0 * float(self.k_x[0, 0]) / self.grid.dx[0]
        )
        self._boundary_indices = np.arange(self.grid.ny, dtype=np.int64)
        self._boundary_response = None
        if (
            self.config.surface.reradiation_enabled
            and self.config.surface.radiation_coupling == "boundary_response"
        ):
            boundary_rhs = np.zeros(
                (self.grid.nx * self.grid.ny, self.grid.ny),
                dtype=np.float64,
            )
            boundary_rhs[
                self._boundary_indices,
                self._boundary_indices,
            ] = self.dt * self.grid.dy
            full_response = self.factor.solve(boundary_rhs)
            self._boundary_response = full_response[
                self._boundary_indices,
            ].copy()
        self.X, self.Y = np.meshgrid(
            self.grid.x_centers,
            self.grid.y_centers,
            indexing="ij",
        )

    def properties_from_temperature(
        self,
        temperature: Array,
        *,
        mode: EvalMode = "accepted",
    ) -> tuple[Array, Array]:
        conductivity_x, _, rho_c = self.directional_properties_from_temperature(
            temperature,
            mode=mode,
        )
        return conductivity_x, rho_c

    def directional_properties_from_temperature(
        self,
        temperature: Array,
        *,
        mode: EvalMode = "accepted",
    ) -> tuple[Array, Array, Array]:
        values = np.asarray(temperature, dtype=np.float64)
        if values.shape[-2:] != (self.grid.nx, self.grid.ny):
            raise ValueError(
                "Temperature field must end in the solver's (Nx, Ny) shape."
            )
        conductivity_x = np.empty_like(values)
        conductivity_y = np.empty_like(values)
        rho_c = np.empty_like(values)
        for region_index, model in enumerate(
            (
                self.region_properties.tps,
                self.region_properties.bond,
                self.region_properties.backing,
            )
        ):
            mask = self.grid.region == region_index
            if not np.any(mask):
                continue
            region_values = values[..., mask, :]
            model_k_x = model.conductivity(
                region_values,
                mode=mode,
                direction="x",
            )
            model_k_y = model.conductivity(
                region_values,
                mode=mode,
                direction="y",
            )
            model_rho_c = model.volumetric_heat_capacity(
                region_values,
                mode=mode,
            )
            if region_index == 1:
                if model.is_temperature_dependent:
                    reference_k = float(
                        np.asarray(
                            model.conductivity(
                                model.reference_temperature,
                                mode=mode,
                                direction="x",
                            )
                        )
                    )
                    model_k_x = (
                        model_k_x
                        * self.bond_conductivity[None, :]
                        / reference_k
                    )
                    model_k_y = (
                        model_k_y
                        * self.bond_conductivity[None, :]
                        / reference_k
                    )
                else:
                    model_k_x = np.broadcast_to(
                        self.bond_conductivity,
                        region_values.shape,
                    )
                    model_k_y = model_k_x
                if self.bond_rho_c != self.config.bond.rho_c:
                    model_rho_c = np.full_like(
                        region_values,
                        self.bond_rho_c,
                    )
            conductivity_x[..., mask, :] = model_k_x
            conductivity_y[..., mask, :] = model_k_y
            rho_c[..., mask, :] = model_rho_c
        if (
            not np.isfinite(conductivity_x).all()
            or np.any(conductivity_x <= 0.0)
            or not np.isfinite(conductivity_y).all()
            or np.any(conductivity_y <= 0.0)
        ):
            bad = np.argwhere(
                ~np.isfinite(conductivity_x)
                | (conductivity_x <= 0.0)
                | ~np.isfinite(conductivity_y)
                | (conductivity_y <= 0.0)
            )[0]
            bad_index = tuple(int(v) for v in bad)
            raise ValueError(
                "Non-positive or non-finite directional conductivity at "
                f"cell={bad_index}, T={float(values[bad_index]):.9g} K."
            )
        if not np.isfinite(rho_c).all() or np.any(rho_c <= 0.0):
            raise ValueError("Volumetric heat capacity must remain finite and positive.")
        return conductivity_x, conductivity_y, rho_c

    def enthalpy_from_temperature(
        self,
        temperature: Array,
        *,
        mode: EvalMode = "accepted",
    ) -> Array:
        values = np.asarray(temperature, dtype=np.float64)
        if values.shape[-2:] != (self.grid.nx, self.grid.ny):
            raise ValueError(
                "Temperature field must end in the solver's (Nx, Ny) shape."
            )
        enthalpy = np.empty_like(values)
        for region_index, model in enumerate(
            (
                self.region_properties.tps,
                self.region_properties.bond,
                self.region_properties.backing,
            )
        ):
            mask = self.grid.region == region_index
            if np.any(mask):
                region_values = values[..., mask, :]
                if (
                    region_index == 1
                    and self.bond_rho_c != self.config.bond.rho_c
                ):
                    enthalpy[..., mask, :] = self.bond_rho_c * (
                        region_values - model.reference_temperature
                    )
                else:
                    enthalpy[..., mask, :] = model.enthalpy(
                        region_values,
                        mode=mode,
                    )
        return enthalpy

    def _x_conductances(self) -> Array:
        interface_index = self.grid.nx_tps - 1
        resistance = (
            self.grid.dx[:-1, None] / (2.0 * self.k_x[:-1])
            + self.grid.dx[1:, None] / (2.0 * self.k_x[1:])
        )
        if self.interface_resistance_y is not None:
            resistance = resistance.copy()
            resistance[interface_index] += self.interface_resistance_y
        return self.grid.dy / resistance

    def _y_conductances(self) -> Array:
        resistance = (
            self.grid.dy / (2.0 * self.k_y[:, :-1])
            + self.grid.dy / (2.0 * self.k_y[:, 1:])
        )
        return self.grid.dx[:, None] / resistance

    def _assemble_conduction(self) -> sparse.csr_matrix:
        ny = self.grid.ny
        x_left = np.arange((self.grid.nx - 1) * ny, dtype=np.int64)
        x_right = x_left + ny
        y_left = (
            np.repeat(np.arange(self.grid.nx, dtype=np.int64) * ny, ny - 1)
            + np.tile(np.arange(ny - 1, dtype=np.int64), self.grid.nx)
        )
        y_right = y_left + 1
        left = np.concatenate((x_left, y_left))
        right = np.concatenate((x_right, y_right))
        conductance = np.concatenate((self.g_x.ravel(), self.g_y.ravel()))
        rows = np.column_stack((left, left, right, right)).ravel()
        columns = np.column_stack((left, right, left, right)).ravel()
        values = np.column_stack(
            (-conductance, conductance, conductance, -conductance)
        ).ravel()
        size = self.grid.nx * ny
        return sparse.coo_matrix((values, (rows, columns)), shape=(size, size)).tocsr()

    def _assemble_conduction_reference(self) -> sparse.csr_matrix:
        rows: list[int] = []
        columns: list[int] = []
        values: list[float] = []
        ny = self.grid.ny

        def add_pair(left: int, right: int, value: float) -> None:
            rows.extend((left, left, right, right))
            columns.extend((left, right, left, right))
            values.extend((-value, value, value, -value))

        for i in range(self.grid.nx - 1):
            for j in range(ny):
                add_pair(i * ny + j, (i + 1) * ny + j, float(self.g_x[i, j]))
        for i in range(self.grid.nx):
            for j in range(ny - 1):
                add_pair(i * ny + j, i * ny + j + 1, float(self.g_y[i, j]))
        size = self.grid.nx * ny
        return sparse.coo_matrix(
            (values, (rows, columns)),
            shape=(size, size),
        ).tocsr()

    def _initialize_sparse_pattern(self) -> None:
        size = self.grid.nx * self.grid.ny
        csr_rows = np.repeat(
            np.arange(size, dtype=np.int64),
            np.diff(self.conduction.indptr),
        )
        csr_keys = csr_rows * np.int64(size) + self.conduction.indices
        diagonal_keys = (
            np.arange(size, dtype=np.int64) * np.int64(size)
            + np.arange(size, dtype=np.int64)
        )
        diagonal_slots = np.searchsorted(csr_keys, diagonal_keys)
        if not np.array_equal(csr_keys[diagonal_slots], diagonal_keys):
            raise RuntimeError("Conduction pattern is missing a cell diagonal.")
        self._conduction_indptr = self.conduction.indptr.copy()
        self._conduction_indices = self.conduction.indices.copy()
        self._csr_rows = csr_rows
        self._csr_keys = csr_keys
        self.diag_slot = diagonal_slots
        self.hot_face_diag_slot = diagonal_slots[: self.grid.ny]

        rows, columns, _ = self._conduction_triplets()
        triplet_keys = rows * np.int64(size) + columns
        self._triplet_slots = np.searchsorted(csr_keys, triplet_keys)
        if not np.array_equal(csr_keys[self._triplet_slots], triplet_keys):
            raise RuntimeError("Conduction triplet does not map to the CSR pattern.")

    def _conduction_triplets(self) -> tuple[Array, Array, Array]:
        ny = self.grid.ny
        x_left = np.arange((self.grid.nx - 1) * ny, dtype=np.int64)
        x_right = x_left + ny
        y_left = (
            np.repeat(np.arange(self.grid.nx, dtype=np.int64) * ny, ny - 1)
            + np.tile(np.arange(ny - 1, dtype=np.int64), self.grid.nx)
        )
        y_right = y_left + 1
        left = np.concatenate((x_left, y_left))
        right = np.concatenate((x_right, y_right))
        conductance = np.concatenate((self.g_x.ravel(), self.g_y.ravel()))
        rows = np.column_stack((left, left, right, right)).ravel()
        columns = np.column_stack((left, right, left, right)).ravel()
        values = np.column_stack(
            (-conductance, conductance, conductance, -conductance)
        ).ravel()
        return rows, columns, values

    def _conduction_data(self) -> Array:
        _, _, values = self._conduction_triplets()
        return np.bincount(
            self._triplet_slots,
            weights=values,
            minlength=len(self._conduction_indices),
        )

    def _assemble_conduction_from_pattern(self) -> sparse.csr_matrix:
        size = self.grid.nx * self.grid.ny
        return sparse.csr_matrix(
            (
                self._conduction_data(),
                self._conduction_indices.copy(),
                self._conduction_indptr.copy(),
            ),
            shape=(size, size),
        )

    def _apply_rear_dirichlet(self) -> None:
        size = self.grid.nx * self.grid.ny
        self._rear_forcing = np.zeros(size, dtype=np.float64)
        if self.rear_temperature is None:
            return
        rear_indices = (
            (self.grid.nx - 1) * self.grid.ny
            + np.arange(self.grid.ny, dtype=np.int64)
        )
        conductance = (
            2.0
            * self.k_x[-1]
            * self.grid.dy
            / self.grid.dx[-1]
        )
        diagonal = np.zeros(size, dtype=np.float64)
        diagonal[rear_indices] = -conductance
        self.conduction = self.conduction + sparse.diags(diagonal)
        self._rear_forcing[rear_indices] = (
            conductance * self.rear_temperature
        )

    def _conduction_flux(self, temperature: Array) -> Array:
        return (
            self.conduction @ np.asarray(temperature).reshape(-1)
            + self._rear_forcing
        ).reshape(np.asarray(temperature).shape)

    def _update_property_state(
        self,
        temperature: Array,
        *,
        mode: EvalMode,
    ) -> tuple[Array, Array]:
        conductivity_x, conductivity_y, rho_c = (
            self.directional_properties_from_temperature(
                temperature,
                mode=mode,
            )
        )
        self.k_x[...] = conductivity_x
        self.k_y[...] = conductivity_y
        self.rho_c[...] = rho_c
        self.mass[...] = self.rho_c * self.volume
        self.g_x = self._x_conductances()
        self.g_y = self._y_conductances()
        self.conduction = self._assemble_conduction_from_pattern()
        self._apply_rear_dirichlet()
        return conductivity_x, rho_c

    def _assemble_system(
        self,
        temperature: Array,
        *,
        mode: EvalMode = "iterate",
        h_eff: Array | None = None,
    ) -> sparse.csc_matrix:
        _, rho_c = self._update_property_state(temperature, mode=mode)
        capacitance = (self.volume * rho_c).reshape(-1)
        matrix = (
            sparse.diags(capacitance)
            - 0.5 * self.dt * self.conduction
        ).tocsr()
        if h_eff is not None:
            radiation_diagonal = np.zeros(self.grid.nx * self.grid.ny)
            radiation_diagonal[self._boundary_indices] = (
                0.5 * self.dt * self.grid.dy * h_eff
            )
            matrix = matrix + sparse.diags(radiation_diagonal)
        return matrix.tocsc()

    def _boundary_energy(self, t0: float, t1: float) -> Array:
        if self.flux_integral_fn is not None:
            values = self.flux_integral_fn(self.grid.y_centers, t0, t1)
        else:
            values = self.case.heat_integral(self.grid.y_centers, t0, t1)
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (self.grid.ny,):
            raise ValueError("Boundary flux integral must return shape (Ny,).")
        return values * self.grid.dy

    def _source_values(self, time: float) -> Array:
        if self.source_fn is None:
            return np.zeros((self.grid.nx, self.grid.ny), dtype=np.float64)
        values = np.asarray(self.source_fn(self.X, self.Y, time), dtype=np.float64)
        if values.shape != (self.grid.nx, self.grid.ny):
            raise ValueError("source_fn must return shape (Nx, Ny).")
        return values

    def radiative_heat_flux(self, surface_temperature: Array) -> Array:
        """Return positive-outward reradiative heat flux in W/m²."""
        values = np.asarray(surface_temperature, dtype=np.float64)
        surface = self.config.surface
        if not surface.reradiation_enabled:
            return np.zeros_like(values)
        return (
            surface.emissivity
            * surface.stefan_boltzmann_constant
            * (
                values ** 4
                - surface.radiation_sink_temperature ** 4
            )
        )

    def surface_conductance(
        self,
        hot_cell_temperature: Array,
        *,
        mode: EvalMode = "accepted",
    ) -> Array:
        """Return ``2 k(T_P) / dx[0]`` at the hot half-cell."""
        cell = np.asarray(hot_cell_temperature, dtype=np.float64)
        if not self.region_properties.tps.is_temperature_dependent:
            return np.full_like(
                cell,
                self.surface_heat_transfer_coefficient,
            )
        conductivity = self.region_properties.tps.conductivity(
            cell,
            mode=mode,
            direction="x",
        )
        return 2.0 * conductivity / self.grid.dx[0]

    def _surface_state(
        self,
        hot_cell_temperature: Array,
        incident_heat_flux: Array,
        *,
        mode: EvalMode,
        step_index: int = -1,
        time: float = float("nan"),
        outer_tolerance: float | None = None,
    ) -> tuple[Array, Array, Array]:
        cell = np.asarray(hot_cell_temperature, dtype=np.float64)
        incident = np.asarray(incident_heat_flux, dtype=np.float64)
        cell, incident = np.broadcast_arrays(cell, incident)
        if not self.config.surface.reradiation_enabled:
            return (
                cell.copy(),
                incident.copy(),
                np.zeros_like(cell),
            )

        coefficient = self.surface_conductance(cell, mode=mode)
        surface = self.config.surface
        sink = surface.radiation_sink_temperature
        radiation_scale = (
            surface.emissivity * surface.stefan_boltzmann_constant
        )
        result = np.empty_like(cell)
        q_net = np.empty_like(cell)
        h_eff = np.empty_like(cell)
        flat_cell = cell.reshape(-1)
        flat_incident = incident.reshape(-1)
        flat_coefficient = coefficient.reshape(-1)
        flat_result = result.reshape(-1)
        flat_q_net = q_net.reshape(-1)
        flat_h_eff = h_eff.reshape(-1)
        incident_norm = float(np.max(np.abs(incident))) if incident.size else 0.0

        for lateral_index, (cell_j, incident_j, coefficient_j) in enumerate(
            zip(flat_cell, flat_incident, flat_coefficient, strict=True)
        ):
            lower = min(sink, cell_j)
            upper = cell_j + (
                max(0.0, incident_j) + radiation_scale * sink ** 4
            ) / coefficient_j
            upper = max(upper, cell_j, sink)

            def residual(value: float) -> float:
                return (
                    coefficient_j * (value - cell_j)
                    - incident_j
                    + radiation_scale * (value ** 4 - sink ** 4)
                )

            lower_residual = residual(lower)
            upper_residual = residual(upper)
            expansion = max(1.0, upper - lower)
            for _ in range(20):
                if lower_residual <= 0.0 <= upper_residual:
                    break
                if lower_residual > 0.0:
                    lower = max(1.0e-12, lower - expansion)
                    lower_residual = residual(lower)
                if upper_residual < 0.0:
                    upper += expansion
                    upper_residual = residual(upper)
                expansion *= 2.0
            else:
                raise RuntimeError(
                    "Could not bracket hot-face surface solve at "
                    f"step={step_index}, t={time:.9g}, j={lateral_index}, "
                    f"T_P={cell_j:.9g} K, q_inc={incident_j:.9g} W/m²."
                )

            value = min(
                max(cell_j + incident_j / coefficient_j, lower),
                upper,
            )
            rho_cp_j = float(
                np.asarray(
                    self.region_properties.tps.volumetric_heat_capacity(
                        cell_j,
                        mode=mode,
                    )
                )
            )
            eta_outer = (
                self.config.solver.nonlinear.residual_temperature_tolerance
                if outer_tolerance is None
                else outer_tolerance
            )
            q_tol_from_outer = (
                0.01
                * eta_outer
                * self.volume[0, 0]
                * rho_cp_j
                / (0.5 * self.dt * self.grid.dy)
            )
            q_tol_abs = 1.0e-6 * max(1.0, incident_norm)
            tolerance = max(
                surface.nonlinear_absolute_flux_tolerance,
                surface.nonlinear_relative_tolerance
                * max(1.0, abs(incident_j)),
                q_tol_abs,
                q_tol_from_outer,
            )
            final_residual = residual(value)
            for _ in range(min(50, surface.nonlinear_max_iterations)):
                if (
                    abs(final_residual) <= tolerance
                    or upper - lower <= 1.0e-10
                ):
                    break
                if final_residual > 0.0:
                    upper = value
                else:
                    lower = value
                derivative = coefficient_j + 4.0 * radiation_scale * value ** 3
                candidate = value - final_residual / derivative
                if not lower < candidate < upper:
                    candidate = 0.5 * (lower + upper)
                value = candidate
                final_residual = residual(value)
            else:
                raise RuntimeError(
                    "Hot-face surface solve did not converge: "
                    f"step={step_index}, t={time:.9g}, j={lateral_index}, "
                    f"T_P={cell_j:.9g} K, q_inc={incident_j:.9g} W/m², "
                    f"bracket=[{lower:.9g}, {upper:.9g}] K, "
                    f"|R_s|={abs(final_residual):.9g} W/m²."
                )
            flat_result[lateral_index] = value
            flat_q_net[lateral_index] = coefficient_j * (value - cell_j)
            radiation_derivative = 4.0 * radiation_scale * value ** 3
            flat_h_eff[lateral_index] = (
                coefficient_j
                * radiation_derivative
                / (coefficient_j + radiation_derivative)
            )
        return result, q_net, h_eff

    def surface_temperature(
        self,
        hot_cell_temperature: Array,
        incident_heat_flux: Array,
    ) -> Array:
        """Reconstruct the boundary temperature from its nonlinear energy balance."""
        temperature, _, _ = self._surface_state(
            hot_cell_temperature,
            incident_heat_flux,
            mode="accepted",
        )
        return temperature

    def _radiative_net_flux(
        self,
        zero_flux_next_hot_cell: Array,
        incident_average_flux: Array,
        incident_next_flux: Array,
        old_radiative_flux: Array,
        initial_guess: Array,
    ) -> tuple[Array, Array, int]:
        """Solve the endpoint-trapezoid radiation boundary with a dense response."""
        if self._boundary_response is None:
            raise RuntimeError("Radiative boundary response was not initialized.")
        zero_next = np.asarray(zero_flux_next_hot_cell, dtype=np.float64)
        incident = np.asarray(incident_average_flux, dtype=np.float64)
        incident_next = np.asarray(incident_next_flux, dtype=np.float64)
        old_radiation = np.asarray(old_radiative_flux, dtype=np.float64)
        net_flux = np.asarray(initial_guess, dtype=np.float64).copy()
        if net_flux.shape != (self.grid.ny,):
            net_flux = incident.copy()
        surface = self.config.surface

        for iteration in range(1, surface.nonlinear_max_iterations + 1):
            # Some accelerated BLAS builds leak benign floating-point status
            # flags from prior fourth-power evaluations into matmul warnings.
            # Check the product explicitly instead of exposing those stale flags.
            with np.errstate(all="ignore"):
                boundary_response = self._boundary_response @ net_flux
            if not np.isfinite(boundary_response).all():
                raise RuntimeError(
                    "Non-finite boundary response during radiation iteration."
                )
            next_hot_cell = zero_next + boundary_response
            next_surface = self.surface_temperature(
                next_hot_cell,
                incident_next,
            )
            target = incident - 0.5 * (
                old_radiation + self.radiative_heat_flux(next_surface)
            )
            residual = target - net_flux
            scale = max(
                1.0,
                float(np.max(np.abs(incident))),
                float(np.max(np.abs(target))),
                float(np.max(np.abs(net_flux))),
            )
            tolerance = (
                surface.nonlinear_absolute_flux_tolerance
                + surface.nonlinear_relative_tolerance * scale
            )
            if float(np.max(np.abs(residual))) <= tolerance:
                return target, next_surface, iteration
            net_flux += surface.nonlinear_relaxation * residual
        raise RuntimeError(
            "Incident/reradiation boundary iteration did not converge within "
            f"{surface.nonlinear_max_iterations} iterations; last flux residual "
            f"was {float(np.max(np.abs(residual))):.6g} W/m²."
        )

    def solve(
        self,
        case: SimulationCase | None = None,
        save_times: Array | list[float] | None = None,
        *,
        initial_temperature: float | Array | None = None,
    ) -> Trajectory:
        if self._use_linear_step:
            return self._solve_linear(
                case=case,
                save_times=save_times,
                initial_temperature=initial_temperature,
            )
        return self._solve_nonlinear(
            case=case,
            save_times=save_times,
            initial_temperature=initial_temperature,
        )

    def _solve_linear(
        self,
        case: SimulationCase | None = None,
        save_times: Array | list[float] | None = None,
        *,
        initial_temperature: float | Array | None = None,
    ) -> Trajectory:
        if case is not None and case != self.case:
            raise ValueError("A solver instance is bound to the case used for assembly.")
        times = np.asarray(
            self.config.time.saved_times() if save_times is None else save_times,
            dtype=np.float64,
        )
        if times.ndim != 1 or len(times) == 0 or not np.isclose(times[0], 0.0):
            raise ValueError("save_times must be a 1D sequence beginning at zero.")
        for time in times:
            self.config.time.assert_aligned(float(time))
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("save_times must be strictly increasing.")
        final_time = float(times[-1])
        n_steps = int(round(final_time / self.dt))
        save_indices = np.rint(times / self.dt).astype(int)
        save_lookup = {int(step): index for index, step in enumerate(save_indices)}

        initial = self.config.initial_temperature if initial_temperature is None else initial_temperature
        if np.asarray(initial).ndim == 0:
            temperature = np.full(
                (self.grid.nx, self.grid.ny),
                float(initial),
                dtype=np.float64,
            )
        else:
            temperature = np.asarray(initial, dtype=np.float64).copy()
            if temperature.shape != (self.grid.nx, self.grid.ny):
                raise ValueError("Initial temperature must be scalar or shape (Nx, Ny).")

        saved = np.empty((len(times), self.grid.nx, self.grid.ny), dtype=np.float64)
        saved[0] = temperature
        saved_surface = np.empty((len(times), self.grid.ny), dtype=np.float64)
        initial_incident = self.case.heat_flux(self.grid.y_centers, 0.0)
        current_surface = self.surface_temperature(
            temperature[0],
            initial_incident,
        )
        saved_surface[0] = current_surface
        initial_energy = float(np.sum(self.mass * temperature))
        energy_times = np.arange(n_steps + 1, dtype=np.float64) * self.dt
        internal = np.empty(n_steps + 1, dtype=np.float64)
        expected = np.empty(n_steps + 1, dtype=np.float64)
        boundary_input_history = np.zeros(n_steps + 1, dtype=np.float64)
        radiated_history = np.zeros(n_steps + 1, dtype=np.float64)
        net_boundary_history = np.zeros(n_steps + 1, dtype=np.float64)
        boundary_energy_increment = np.zeros(
            (n_steps, self.grid.ny),
            dtype=np.float64,
        )
        nonlinear_iterations = np.zeros(n_steps, dtype=np.int32)
        internal[0] = initial_energy
        expected[0] = initial_energy
        boundary_input = 0.0
        radiated = 0.0
        net_boundary = 0.0
        source_energy = 0.0
        minimum = float(np.min(temperature))
        maximum_cell = float(np.max(temperature))
        maximum_hot_face = float(np.max(current_surface))
        previous_net_flux = np.zeros(self.grid.ny, dtype=np.float64)
        linear_solve_count = 0

        flat = temperature.reshape(-1)
        for step in range(n_steps):
            t0 = step * self.dt
            t1 = (step + 1) * self.dt
            boundary_energy = self._boundary_energy(t0, t1)
            source0 = self._source_values(t0)
            source1 = self._source_values(t1)
            source_average = 0.5 * (source0 + source1)
            load = np.zeros((self.grid.nx, self.grid.ny), dtype=np.float64)
            load += source_average * self.volume
            rhs_without_boundary = (
                self.matrix_b @ flat
                + self.dt * load.reshape(-1)
            )
            if self.config.surface.reradiation_enabled:
                zero_flux_next = self.factor.solve(rhs_without_boundary)
                linear_solve_count += 1
                incident_average = (
                    boundary_energy / (self.dt * self.grid.dy)
                )
                (
                    net_flux,
                    _,
                    iteration_count,
                ) = self._radiative_net_flux(
                    zero_flux_next[self._boundary_indices],
                    incident_average,
                    self.case.heat_flux(self.grid.y_centers, t1),
                    self.radiative_heat_flux(current_surface),
                    previous_net_flux,
                )
                rhs = rhs_without_boundary.copy()
                rhs[self._boundary_indices] += (
                    self.dt * self.grid.dy * net_flux
                )
                flat = self.factor.solve(rhs)
                linear_solve_count += 1
                previous_net_flux = net_flux
                nonlinear_iterations[step] = iteration_count
                step_input = float(np.sum(boundary_energy))
                step_net = (
                    self.dt * self.grid.dy * float(np.sum(net_flux))
                )
                boundary_energy_increment[step] = (
                    self.dt * self.grid.dy * net_flux
                )
                step_radiated = step_input - step_net
            else:
                load[0] += boundary_energy / self.dt
                rhs = self.matrix_b @ flat + self.dt * load.reshape(-1)
                flat = self.factor.solve(rhs)
                linear_solve_count += 1
                step_input = float(np.sum(boundary_energy))
                step_radiated = 0.0
                step_net = step_input
                boundary_energy_increment[step] = boundary_energy
            temperature = flat.reshape(self.grid.nx, self.grid.ny)
            minimum = min(minimum, float(np.min(temperature)))
            maximum_cell = max(maximum_cell, float(np.max(temperature)))
            current_surface = self.surface_temperature(
                temperature[0],
                self.case.heat_flux(self.grid.y_centers, t1),
            )
            maximum_hot_face = max(
                maximum_hot_face,
                float(np.max(current_surface)),
            )

            boundary_input += step_input
            radiated += step_radiated
            net_boundary += step_net
            source_energy += self.dt * float(np.sum(source_average * self.volume))
            internal[step + 1] = float(np.sum(self.mass * temperature))
            expected[step + 1] = (
                initial_energy + net_boundary + source_energy
            )
            boundary_input_history[step + 1] = boundary_input
            radiated_history[step + 1] = radiated
            net_boundary_history[step + 1] = net_boundary
            if step + 1 in save_lookup:
                saved[save_lookup[step + 1]] = temperature
                saved_surface[save_lookup[step + 1]] = current_surface

        residual = internal - expected
        return Trajectory(
            times=times,
            x_centers=self.grid.x_centers.copy(),
            y_centers=self.grid.y_centers.copy(),
            dx=self.grid.dx.copy(),
            temperatures=saved,
            surface_temperatures=saved_surface,
            energy_times=energy_times,
            internal_energy=internal,
            expected_energy=expected,
            energy_residual=residual,
            boundary_input_energy=boundary_input_history,
            radiated_energy=radiated_history,
            net_boundary_energy=net_boundary_history,
            nonlinear_iteration_counts=nonlinear_iterations,
            minimum_temperature=minimum,
            maximum_hot_face_temperature=maximum_hot_face,
            factorization_count=self.factorization_count,
            step_driver="linear",
            property_model=self.config.validity.tps_property_model,
            linear_solve_count=linear_solve_count,
            max_nonlinear_iterations=(
                int(np.max(nonlinear_iterations))
                if nonlinear_iterations.size
                else 0
            ),
            iteration_range_clamps=0,
            accepted_range_excursions=0,
            solver_converged=True,
            boundary_energy_increment=boundary_energy_increment,
            observed_temperature_range=(minimum, maximum_cell),
            property_query_temperature_range=(minimum, maximum_cell),
        )

    def _nonlinear_residual(
        self,
        candidate: Array,
        old_temperature: Array,
        old_enthalpy: Array,
        old_conduction_flux: Array,
        source_increment: Array,
        incident_energy: Array,
        incident_next: Array,
        old_radiation: Array,
        *,
        step_index: int,
        time: float,
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        enthalpy = self.enthalpy_from_temperature(
            candidate,
            mode="iterate",
        )
        _, rho_c = self._update_property_state(candidate, mode="iterate")
        conduction_flux = self._conduction_flux(candidate)
        surface_temperature, q_net, h_eff = self._surface_state(
            candidate[0],
            incident_next,
            mode="iterate",
            step_index=step_index,
            time=time,
            outer_tolerance=(
                self.config.solver.nonlinear.residual_temperature_tolerance
            ),
        )
        next_radiation = self.radiative_heat_flux(surface_temperature)
        boundary_increment = incident_energy - (
            0.5
            * self.dt
            * self.grid.dy
            * (old_radiation + next_radiation)
        )
        residual = (
            self.volume * (enthalpy - old_enthalpy)
            - 0.5
            * self.dt
            * (conduction_flux + old_conduction_flux)
            - source_increment
        )
        residual = residual.copy()
        residual[0] -= boundary_increment
        if not np.isfinite(residual).all():
            bad = np.argwhere(~np.isfinite(residual))[0]
            raise RuntimeError(
                "Non-finite nonlinear residual at "
                f"step={step_index}, t={time:.9g}, "
                f"cell={tuple(int(v) for v in bad)}."
            )
        return (
            residual,
            rho_c,
            surface_temperature,
            q_net,
            h_eff,
            boundary_increment,
        )

    def _table_range_damping(
        self,
        temperature: Array,
        update: Array,
        damping: float,
    ) -> float:
        limited = float(damping)
        for region_index, model in enumerate(
            (
                self.region_properties.tps,
                self.region_properties.bond,
                self.region_properties.backing,
            )
        ):
            table = getattr(model, "table", None)
            if table is None:
                continue
            mask = self.grid.region == region_index
            values = temperature[mask]
            direction = update[mask]
            positive = direction > 0.0
            negative = direction < 0.0
            candidates: list[float] = []
            if np.any(positive):
                candidates.append(
                    float(
                        np.min(
                            (table.temperature_max - values[positive])
                            / direction[positive]
                        )
                    )
                )
            if np.any(negative):
                candidates.append(
                    float(
                        np.min(
                            (table.temperature_min - values[negative])
                            / direction[negative]
                        )
                    )
                )
            positive_candidates = [value for value in candidates if value > 0.0]
            if positive_candidates:
                limited = min(limited, 0.99 * min(positive_candidates))
        return max(
            self.config.solver.nonlinear.minimum_damping,
            limited,
        )

    def _solve_nonlinear(
        self,
        case: SimulationCase | None = None,
        save_times: Array | list[float] | None = None,
        *,
        initial_temperature: float | Array | None = None,
    ) -> Trajectory:
        if case is not None and case != self.case:
            raise ValueError("A solver instance is bound to the case used for assembly.")
        times = np.asarray(
            self.config.time.saved_times() if save_times is None else save_times,
            dtype=np.float64,
        )
        if times.ndim != 1 or len(times) == 0 or not np.isclose(times[0], 0.0):
            raise ValueError("save_times must be a 1D sequence beginning at zero.")
        for time in times:
            self.config.time.assert_aligned(float(time))
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("save_times must be strictly increasing.")
        n_steps = int(round(float(times[-1]) / self.dt))
        save_indices = np.rint(times / self.dt).astype(int)
        save_lookup = {int(step): index for index, step in enumerate(save_indices)}

        initial = (
            self.config.initial_temperature
            if initial_temperature is None
            else initial_temperature
        )
        if np.asarray(initial).ndim == 0:
            temperature = np.full(
                (self.grid.nx, self.grid.ny),
                float(initial),
                dtype=np.float64,
            )
        else:
            temperature = np.asarray(initial, dtype=np.float64).copy()
            if temperature.shape != (self.grid.nx, self.grid.ny):
                raise ValueError("Initial temperature must be scalar or shape (Nx, Ny).")

        saved = np.empty(
            (len(times), self.grid.nx, self.grid.ny),
            dtype=np.float64,
        )
        saved[0] = temperature
        saved_surface = np.empty((len(times), self.grid.ny), dtype=np.float64)
        incident_old = self.case.heat_flux(self.grid.y_centers, 0.0)
        current_surface, _, _ = self._surface_state(
            temperature[0],
            incident_old,
            mode="accepted",
            step_index=0,
            time=0.0,
        )
        old_radiation = self.radiative_heat_flux(current_surface)
        saved_surface[0] = current_surface

        initial_enthalpy = self.enthalpy_from_temperature(
            temperature,
            mode="accepted",
        )
        initial_energy = float(np.sum(self.volume * initial_enthalpy))
        energy_times = np.arange(n_steps + 1, dtype=np.float64) * self.dt
        internal = np.empty(n_steps + 1, dtype=np.float64)
        expected = np.empty(n_steps + 1, dtype=np.float64)
        boundary_input_history = np.zeros(n_steps + 1, dtype=np.float64)
        radiated_history = np.zeros(n_steps + 1, dtype=np.float64)
        net_boundary_history = np.zeros(n_steps + 1, dtype=np.float64)
        boundary_energy_increment = np.zeros(
            (n_steps, self.grid.ny),
            dtype=np.float64,
        )
        nonlinear_iterations = np.zeros(n_steps, dtype=np.int32)
        nonlinear_final_residuals = np.zeros(n_steps, dtype=np.float64)
        nonlinear_damped_iterations = np.zeros(n_steps, dtype=np.int32)
        nonlinear_backtracks = np.zeros(n_steps, dtype=np.int32)
        internal[0] = initial_energy
        expected[0] = initial_energy
        boundary_input = 0.0
        radiated = 0.0
        net_boundary = 0.0
        source_energy = 0.0
        observed_minimum = float(np.min(temperature))
        observed_maximum = float(np.max(temperature))
        maximum_hot_face = float(np.max(current_surface))
        controls = self.config.solver.nonlinear
        linear_solve_count = 0

        for step in range(n_steps):
            t0 = step * self.dt
            t1 = (step + 1) * self.dt
            incident_energy = self._boundary_energy(t0, t1)
            incident_next = self.case.heat_flux(self.grid.y_centers, t1)
            source_average = 0.5 * (
                self._source_values(t0) + self._source_values(t1)
            )
            source_increment = self.dt * source_average * self.volume

            old_enthalpy = self.enthalpy_from_temperature(
                temperature,
                mode="accepted",
            )
            self._update_property_state(temperature, mode="accepted")
            old_conduction_flux = self._conduction_flux(temperature)
            capacitance = (self.volume * self.rho_c).reshape(-1)
            frozen_matrix = (
                sparse.diags(capacitance)
                - 0.5 * self.dt * self.conduction
            ).tocsc()
            frozen_factor = splu(frozen_matrix)
            self.factorization_count += 1
            frozen_boundary = incident_energy - (
                self.dt * self.grid.dy * old_radiation
            )
            frozen_rhs = (
                capacitance * temperature.reshape(-1)
                + 0.5 * self.dt * old_conduction_flux.reshape(-1)
                + source_increment.reshape(-1)
            )
            frozen_rhs = frozen_rhs.copy()
            frozen_rhs[self._boundary_indices] += frozen_boundary
            candidate = frozen_factor.solve(frozen_rhs).reshape(
                temperature.shape
            )
            linear_solve_count += 1

            damping_history: list[float] = []
            step_backtracks = 0
            last_residual_temperature = float("inf")
            last_update = float("inf")
            last_worst_cell = (0, 0)
            accepted_state: tuple[
                Array,
                Array,
                Array,
                Array,
                Array,
                Array,
            ] | None = None
            for iteration in range(1, controls.max_iterations + 1):
                state = self._nonlinear_residual(
                    candidate,
                    temperature,
                    old_enthalpy,
                    old_conduction_flux,
                    source_increment,
                    incident_energy,
                    incident_next,
                    old_radiation,
                    step_index=step,
                    time=t1,
                )
                residual, _, _, _, h_eff, _ = state
                base_norm = float(np.max(np.abs(residual)))
                jacobian = self._assemble_system(
                    candidate,
                    mode="iterate",
                    h_eff=h_eff,
                )
                factor = splu(jacobian)
                self.factorization_count += 1
                delta = factor.solve(-residual.reshape(-1)).reshape(
                    candidate.shape
                )
                linear_solve_count += 1
                delta_norm = float(np.max(np.abs(delta)))
                damping = controls.initial_damping
                if delta_norm > 0.0:
                    damping = min(
                        damping,
                        controls.max_temperature_step / delta_norm,
                    )
                damping = self._table_range_damping(
                    candidate,
                    delta,
                    damping,
                )
                damping = max(controls.minimum_damping, damping)

                trial_state = None
                trial = candidate
                for backtrack in range(controls.max_backtracks + 1):
                    trial = candidate + damping * delta
                    trial_state = self._nonlinear_residual(
                        trial,
                        temperature,
                        old_enthalpy,
                        old_conduction_flux,
                        source_increment,
                        incident_energy,
                        incident_next,
                        old_radiation,
                        step_index=step,
                        time=t1,
                    )
                    trial_norm = float(
                        np.max(np.abs(trial_state[0]))
                    )
                    if (
                        trial_norm
                        <= (1.0 - controls.armijo_c * damping) * base_norm
                        or base_norm
                        <= np.finfo(np.float64).eps
                    ):
                        break
                    if backtrack == controls.max_backtracks:
                        break
                    step_backtracks += 1
                    damping = max(
                        controls.minimum_damping,
                        0.5 * damping,
                    )
                assert trial_state is not None
                damping_history.append(float(damping))
                if damping < 1.0 - 16.0 * np.finfo(np.float64).eps:
                    nonlinear_damped_iterations[step] += 1
                applied_update = damping * delta
                candidate = trial
                residual, rho_c, _, _, _, _ = trial_state
                temperature_residual = np.abs(residual) / (
                    self.volume * rho_c
                )
                worst_flat = int(np.argmax(temperature_residual))
                last_worst_cell = np.unravel_index(
                    worst_flat,
                    candidate.shape,
                )
                last_residual_temperature = float(
                    temperature_residual[last_worst_cell]
                )
                last_update = float(np.max(np.abs(applied_update)))
                update_limit = (
                    controls.update_temperature_tolerance
                    + controls.update_relative_tolerance
                    * float(np.max(np.abs(candidate)))
                )
                if (
                    last_residual_temperature
                    <= controls.residual_temperature_tolerance
                    and last_update <= update_limit
                ):
                    accepted_state = trial_state
                    nonlinear_iterations[step] = iteration
                    nonlinear_final_residuals[step] = (
                        last_residual_temperature
                    )
                    nonlinear_backtracks[step] = step_backtracks
                    break
            if accepted_state is None:
                raise RuntimeError(
                    "Global nonlinear solve did not converge: "
                    f"step={step}, t={t1:.9g}, "
                    f"||r||_inf={last_residual_temperature:.9g} K, "
                    f"||delta||_inf={last_update:.9g} K, "
                    f"lambda_history={damping_history}, "
                    f"worst_cell={tuple(int(v) for v in last_worst_cell)}."
                )
            if (
                np.ptp(temperature) == 0.0
                and not np.any(source_increment)
                and not np.any(incident_energy)
                and not np.any(old_radiation)
            ):
                # A uniform adiabatic state is an exact discrete fixed point.
                # Remove only the sparse-solve roundoff from that invariant.
                candidate = temperature.copy()

            # Enforce the configured extrapolation policy only on the state
            # that is actually accepted and saved.
            self._update_property_state(candidate, mode="accepted")
            accepted_enthalpy = self.enthalpy_from_temperature(
                candidate,
                mode="accepted",
            )
            current_surface, _, _ = self._surface_state(
                candidate[0],
                incident_next,
                mode="accepted",
                step_index=step,
                time=t1,
            )
            next_radiation = self.radiative_heat_flux(current_surface)
            accepted_boundary = incident_energy - (
                0.5
                * self.dt
                * self.grid.dy
                * (old_radiation + next_radiation)
            )
            boundary_energy_increment[step] = accepted_boundary
            step_input = float(np.sum(incident_energy))
            step_radiated = float(
                np.sum(
                    0.5
                    * self.dt
                    * self.grid.dy
                    * (old_radiation + next_radiation)
                )
            )
            step_net = float(np.sum(accepted_boundary))
            step_source = float(np.sum(source_increment))

            temperature = candidate
            incident_old = incident_next
            old_radiation = next_radiation
            observed_minimum = min(
                observed_minimum,
                float(np.min(temperature)),
            )
            observed_maximum = max(
                observed_maximum,
                float(np.max(temperature)),
            )
            maximum_hot_face = max(
                maximum_hot_face,
                float(np.max(current_surface)),
            )
            boundary_input += step_input
            radiated += step_radiated
            net_boundary += step_net
            source_energy += step_source
            internal[step + 1] = float(
                np.sum(self.volume * accepted_enthalpy)
            )
            expected[step + 1] = (
                initial_energy + net_boundary + source_energy
            )
            boundary_input_history[step + 1] = boundary_input
            radiated_history[step + 1] = radiated
            net_boundary_history[step + 1] = net_boundary
            if step + 1 in save_lookup:
                saved[save_lookup[step + 1]] = temperature
                saved_surface[save_lookup[step + 1]] = current_surface

        return Trajectory(
            times=times,
            x_centers=self.grid.x_centers.copy(),
            y_centers=self.grid.y_centers.copy(),
            dx=self.grid.dx.copy(),
            temperatures=saved,
            surface_temperatures=saved_surface,
            energy_times=energy_times,
            internal_energy=internal,
            expected_energy=expected,
            energy_residual=internal - expected,
            boundary_input_energy=boundary_input_history,
            radiated_energy=radiated_history,
            net_boundary_energy=net_boundary_history,
            nonlinear_iteration_counts=nonlinear_iterations,
            minimum_temperature=observed_minimum,
            maximum_hot_face_temperature=maximum_hot_face,
            factorization_count=self.factorization_count,
            step_driver="nonlinear",
            property_model=self.config.validity.tps_property_model,
            linear_solve_count=linear_solve_count,
            max_nonlinear_iterations=(
                int(np.max(nonlinear_iterations))
                if nonlinear_iterations.size
                else 0
            ),
            iteration_range_clamps=(
                self.region_properties.iteration_range_clamps
            ),
            accepted_range_excursions=(
                self.region_properties.accepted_range_excursions
            ),
            solver_converged=True,
            nonlinear_final_residual_norms=nonlinear_final_residuals,
            nonlinear_damped_iteration_counts=nonlinear_damped_iterations,
            nonlinear_backtrack_counts=nonlinear_backtracks,
            failed_nonlinear_iterations=0,
            property_query_temperature_range=(
                self.region_properties.query_temperature_range
            ),
            boundary_energy_increment=boundary_energy_increment,
            observed_temperature_range=(observed_minimum, observed_maximum),
        )

    def face_flux_and_traces(
        self,
        temperatures: Array,
        face_index: int,
        *,
        conductivity: Array | None = None,
    ) -> tuple[Array, Array, Array]:
        values = np.asarray(temperatures, dtype=np.float64)
        if values.shape[-2:] != (self.grid.nx, self.grid.ny):
            raise ValueError("Temperature field has incompatible spatial shape.")
        if not 0 <= face_index < self.grid.nx - 1:
            raise ValueError("face_index is out of range.")
        left = values[..., face_index, :]
        right = values[..., face_index + 1, :]
        area = self.grid.dy
        if (
            conductivity is None
            and not self.region_properties.is_temperature_dependent
        ):
            # Preserve the legacy constant-property arithmetic verbatim.
            flux = (self.g_x[face_index] / area) * (left - right)
            left_trace = left - flux * self.grid.dx[face_index] / (2.0 * self.k[face_index])
            right_trace = right + flux * self.grid.dx[face_index + 1] / (
                2.0 * self.k[face_index + 1]
            )
            return flux, left_trace, right_trace
        if conductivity is None:
            conductivity, _ = self.properties_from_temperature(
                values,
                mode="accepted",
            )
        else:
            conductivity = np.asarray(conductivity, dtype=np.float64)
            if conductivity.shape != values.shape:
                raise ValueError("conductivity must have the temperature-field shape.")
        left_k = conductivity[..., face_index, :]
        right_k = conductivity[..., face_index + 1, :]
        resistance = (
            self.grid.dx[face_index] / (2.0 * left_k)
            + self.grid.dx[face_index + 1] / (2.0 * right_k)
        )
        if (
            self.interface_resistance_y is not None
            and face_index == self.grid.nx_tps - 1
        ):
            resistance = resistance + self.interface_resistance_y
        flux = (left - right) / resistance
        left_trace = left - flux * self.grid.dx[face_index] / (2.0 * left_k)
        right_trace = right + flux * self.grid.dx[face_index + 1] / (
            2.0 * right_k
        )
        return flux, left_trace, right_trace

    def structural_interface_temperature(self, temperatures: Array) -> Array:
        _, _, right_trace = self.face_flux_and_traces(
            temperatures,
            self.grid.structural_face_index,
        )
        return right_trace

    def bond_temperature_envelope(self, temperatures: Array) -> Array:
        """Maximum bond temperature by time and lateral coordinate.

        The bond design limit applies to the material, including its two
        interfaces. Using only cell centres makes the reported maximum move
        upward as the through-thickness mesh is refined because the hottest
        bond centre moves closer to the TPS interface. Reconstructed face
        traces remove that sampling artifact while retaining an interior-cell
        maximum for completeness.
        """
        values = np.asarray(temperatures, dtype=np.float64)
        if values.ndim < 2 or values.shape[-2:] != (
            self.grid.nx,
            self.grid.ny,
        ):
            raise ValueError(
                "Bond-temperature input must end in shape (Nx, Ny)."
            )
        if not self.grid.nx_bond:
            return np.full(
                values.shape[:-2] + (self.grid.ny,),
                np.nan,
                dtype=np.float64,
            )
        cell_maximum = np.max(
            values[..., self.grid.bond_slice, :],
            axis=-2,
        )
        _, _, hot_side_trace = self.face_flux_and_traces(
            values,
            self.grid.nx_tps - 1,
        )
        _, cold_side_trace, _ = self.face_flux_and_traces(
            values,
            self.grid.structural_face_index,
        )
        return np.maximum.reduce(
            (cell_maximum, hot_side_trace, cold_side_trace)
        )

    def quantities_of_interest(self, temperatures: Array, times: Array) -> dict[str, float]:
        values = np.asarray(temperatures, dtype=np.float64)
        times = np.asarray(times, dtype=np.float64)
        if values.ndim != 3 or values.shape[0] != len(times):
            raise ValueError("QoI input must have shape (Nt, Nx, Ny).")
        if self.grid.nx_bond:
            bond_values = self.bond_temperature_envelope(values)
            bond_flat = int(np.argmax(bond_values))
            bond_index = np.unravel_index(bond_flat, bond_values.shape)
            bond_max = float(bond_values[bond_index])
            bond_time = float(times[bond_index[0]])
            bond_y = float(self.grid.y_centers[bond_index[1]])
        else:
            bond_max = bond_time = bond_y = float("nan")
        interface = self.structural_interface_temperature(values)
        interface_flat = int(np.argmax(interface))
        interface_index = np.unravel_index(interface_flat, interface.shape)
        backing = values[:, self.grid.backing_slice, :]
        backing_index = np.unravel_index(int(np.argmax(backing)), backing.shape)
        return {
            "bond_max": bond_max,
            "bond_peak_time": bond_time,
            "bond_peak_y": bond_y,
            "structural_interface_max": float(interface[interface_index]),
            "structural_interface_peak_time": float(times[interface_index[0]]),
            "structural_interface_peak_y": float(self.grid.y_centers[interface_index[1]]),
            "backing_max": float(backing[backing_index]),
            "backing_peak_time": float(times[backing_index[0]]),
            "backing_peak_y": float(self.grid.y_centers[backing_index[2]]),
        }


def horizon_diagnostics(
    solver: TPSFVSolver,
    trajectory: Trajectory,
    design_window: float | None = None,
    final_steps: int = 5,
    heating_fraction: float = 1e-3,
) -> dict[str, float | bool | int | str]:
    if len(trajectory.times) <= final_steps:
        raise ValueError("Horizon diagnostics require more saved times than final_steps.")
    q_samples = np.stack([
        solver.case.heat_flux(solver.grid.y_centers, float(time))
        for time in trajectory.times
    ])
    q_peak = float(np.max(q_samples))
    q_final = float(np.max(q_samples[-1]))
    qois_bond = np.max(
        solver.bond_temperature_envelope(trajectory.temperatures),
        axis=1,
    )
    interface = np.max(
        solver.structural_interface_temperature(trajectory.temperatures),
        axis=1,
    )
    bond_peak_index = int(np.argmax(qois_bond))
    interface_peak_index = int(np.argmax(interface))
    bond_endpoint = bool(
        np.isclose(qois_bond[-1], np.max(qois_bond), rtol=1e-10, atol=1e-10)
    )
    interface_endpoint = bool(
        np.isclose(interface[-1], np.max(interface), rtol=1e-10, atol=1e-10)
    )
    bond_control = "endpoint" if bond_endpoint else "interior_peak"
    interface_control = "endpoint" if interface_endpoint else "interior_peak"
    requested_window = (
        float(trajectory.times[-1])
        if design_window is None
        else float(design_window)
    )
    design_window_covered = bool(
        trajectory.times[-1] >= requested_window - 1e-10
    )
    heating_ended = q_peak == 0.0 or q_final <= heating_fraction * q_peak

    def tail_trend(values: Array) -> str:
        differences = np.diff(values[-(final_steps + 1):])
        if np.all(differences < 0.0):
            return "declining"
        if np.all(differences > 0.0):
            return "rising"
        if np.all(np.abs(differences) <= 1e-10):
            return "flat"
        return "mixed"

    return {
        "accepted": bool(heating_ended and design_window_covered),
        "heating_ended": bool(heating_ended),
        "design_window_covered": design_window_covered,
        "design_window": requested_window,
        "bond_control": bond_control,
        "structural_interface_control": interface_control,
        "bond_tail_trend": tail_trend(qois_bond),
        "structural_interface_tail_trend": tail_trend(interface),
        "bond_peak_index": bond_peak_index,
        "interface_peak_index": interface_peak_index,
        "q_final_fraction": 0.0 if q_peak == 0.0 else q_final / q_peak,
    }
