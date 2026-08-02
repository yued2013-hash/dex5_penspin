"""Cache-backed and physical first-leg tasks for Isaac Lab training."""

import gymnasium as gym

from . import agents


gym.register(
    id="Dex5-CanonicalReset-Physics-Direct-v0",
    entry_point=(
        f"{__name__}.canonical_reset_physics_env:"
        "Dex5CanonicalResetPhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.canonical_reset_physics_env:"
            "Dex5CanonicalResetPhysicsEnvCfg"
        ),
    },
)


gym.register(
    id="Dex5-S0.2-P3HardwareCompat-Audit-Direct-v0",
    entry_point=(
        f"{__name__}.s0_hardware_compatible_env:"
        "Dex5S0P3HardwareCompatibleAuditEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.s0_hardware_compatible_env:"
            "Dex5S0P3HardwareCompatibleAuditEnvCfg"
        ),
    },
)


gym.register(
    id="Dex5-S0.8-P3MeasuredHardwareCompat-Audit-Direct-v0",
    entry_point=(
        f"{__name__}.s0_hardware_compatible_env:"
        "Dex5S0P3MeasuredHardwareCompatibleAuditEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.s0_hardware_compatible_env:"
            "Dex5S0P3MeasuredHardwareCompatibleAuditEnvCfg"
        ),
    },
)


gym.register(
    id="Dex5-S0-HardwareCompatibleReset-Direct-v0",
    entry_point=(
        f"{__name__}.s0_hardware_compatible_env:"
        "Dex5S0HardwareCompatibleResetEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.s0_hardware_compatible_env:"
            "Dex5S0HardwareCompatibleResetEnvCfg"
        ),
    },
)


for task_id, cfg_name in (
    (
        "Dex5-S0-ShortRotation-15Deg-Direct-v0",
        "Dex5S0ShortRotation15DegEnvCfg",
    ),
    (
        "Dex5-S0-ShortRotation-30Deg-Direct-v0",
        "Dex5S0ShortRotation30DegEnvCfg",
    ),
    (
        "Dex5-S0-ShortRotation-37Deg-Direct-v0",
        "Dex5S0ShortRotation37DegEnvCfg",
    ),
    (
        "Dex5-S0-ShortRotation-35Deg-Direct-v0",
        "Dex5S0ShortRotation35DegEnvCfg",
    ),
    (
        "Dex5-S0-ShortRotation-40Deg-Direct-v0",
        "Dex5S0ShortRotation40DegEnvCfg",
    ),
    (
        "Dex5-S0-ShortRotation-35Deg-Long-Direct-v0",
        "Dex5S0ShortRotation35DegLongEnvCfg",
    ),
    (
        "Dex5-S0-ShortRotation-Long-Direct-v0",
        "Dex5S0ShortRotationLongEnvCfg",
    ),
    (
        "Dex5-S0-ShortRotation-Long-Nominal-Direct-v0",
        "Dex5S0ShortRotationLongNominalEnvCfg",
    ),
    (
        "Dex5-S0-ShortRotation-16-Nominal-Direct-v0",
        "Dex5S0ShortRotation16NominalEnvCfg",
    ),
    (
        "Dex5-S0-ShortRotation-16-MediumDR-Direct-v0",
        "Dex5S0ShortRotation16MediumDREnvCfg",
    ),
    ("Dex5-S0-ShortRotation-Direct-v0", "Dex5S0ShortRotationEnvCfg"),
    (
        "Dex5-S0-ShortRotation-MediumDR-Direct-v0",
        "Dex5S0ShortRotationMediumDREnvCfg",
    ),
):
    gym.register(
        id=task_id,
        entry_point=(
            f"{__name__}.s0_short_rotation_env:Dex5S0ShortRotationEnv"
        ),
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                f"{__name__}.s0_short_rotation_env:{cfg_name}"
            ),
            "skrl_cfg_entry_point": (
                f"{agents.__name__}:skrl_ppo_s0_short_rotation_cfg.yaml"
            ),
        },
    )


_O1_2_ORACLE_CONFIGS = {
    "Dex5-PenSpin-OracleAligned-A-Direct-v0": (
        "Dex5PrivilegedOracleAlignedMixtureACfg"
    ),
    "Dex5-PenSpin-OracleAligned-B-Direct-v0": (
        "Dex5PrivilegedOracleAlignedMixtureBCfg"
    ),
    "Dex5-PenSpin-OracleAligned-P3Only-Direct-v0": (
        "Dex5PrivilegedOracleAlignedP3OnlyCfg"
    ),
    "Dex5-PenSpin-OracleAligned-P3Axis-Direct-v0": (
        "Dex5PrivilegedOracleAlignedP3AxisCfg"
    ),
    "Dex5-PenSpin-Oracle-P3State-Direct-v0": "Dex5PrivilegedOracleP3StateCfg",
    "Dex5-PenSpin-Oracle-P3Temporal-Direct-v0": (
        "Dex5PrivilegedOracleP3TemporalCfg"
    ),
    "Dex5-PenSpin-Oracle-P3EntryCurriculum-Direct-v0": (
        "Dex5PrivilegedOracleP3EntryCurriculumCfg"
    ),
    "Dex5-PenSpin-Oracle-P3SupportMargin-Direct-v0": (
        "Dex5PrivilegedOracleP3SupportMarginCfg"
    ),
    "Dex5-PenSpin-Oracle-P3Recoverability-Direct-v0": (
        "Dex5PrivilegedOracleP3RecoverabilityCfg"
    ),
    "Dex5-PenSpin-Oracle-P3FixedHorizon-Direct-v0": (
        "Dex5PrivilegedOracleP3FixedHorizonCfg"
    ),
    "Dex5-PenSpin-Oracle-P3FixedHorizonTerminal-Direct-v0": (
        "Dex5PrivilegedOracleP3FixedHorizonTerminalCfg"
    ),
    "Dex5-PenSpin-Oracle-P3RecoveryGrace-Direct-v0": (
        "Dex5PrivilegedOracleP3RecoveryGraceCfg"
    ),
}

for task_id, cfg_name in _O1_2_ORACLE_CONFIGS.items():
    gym.register(
        id=task_id,
        entry_point=(
            f"{__name__}.privileged_oracle_physics_env:"
            "Dex5PrivilegedOraclePhysicsEnv"
        ),
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                f"{__name__}.privileged_oracle_physics_env:{cfg_name}"
            ),
            "skrl_cfg_entry_point": (
                f"{agents.__name__}:skrl_ppo_privileged_oracle_cfg.yaml"
            ),
        },
    )

gym.register(
    id="Dex5-PenSpin-PrivilegedOracle-Direct-v0",
    entry_point=(
        f"{__name__}.privileged_oracle_physics_env:"
        "Dex5PrivilegedOraclePhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.privileged_oracle_physics_env:"
            "Dex5PrivilegedOraclePhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_privileged_oracle_cfg.yaml",
    },
)


gym.register(
    id="Dex5-IndexNearSurface-Cache-Direct-v0",
    entry_point=f"{__name__}.index_near_surface_cache_env:Dex5IndexNearSurfaceCacheEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.index_near_surface_cache_env:Dex5IndexNearSurfaceCacheEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Dex5-V9P1-P2-ResetBank-Direct-v0",
    entry_point=(
        f"{__name__}.recapture_p3_physics_env:Dex5RecaptureP3PhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.recapture_p3_physics_env:"
            "Dex5V9P1P2ResetBankPhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml",
    },
)


_P1_P2_TASK_CONFIGS = {
    "Dex5-P1P2-C0-Direct-v0": "Dex5P1P2C0PhysicsEnvCfg",
    "Dex5-P1P2-C1-Direct-v0": "Dex5P1P2C1PhysicsEnvCfg",
    "Dex5-P1P2-C1TeacherResidual-Direct-v0": "Dex5P1P2C1TeacherResidualPhysicsEnvCfg",
    "Dex5-P1P2-C2-Direct-v0": "Dex5P1P2C2PhysicsEnvCfg",
    "Dex5-P1P2-C2TeacherResidual-Direct-v0": "Dex5P1P2C2TeacherResidualPhysicsEnvCfg",
    "Dex5-P1P2-C2StateTeacherResidual-Direct-v0": "Dex5P1P2C2StateTeacherResidualPhysicsEnvCfg",
    "Dex5-P1P2-C3-Direct-v0": "Dex5P1P2C3PhysicsEnvCfg",
    "Dex5-P1P2-C3TeacherResidual-Direct-v0": "Dex5P1P2C3TeacherResidualPhysicsEnvCfg",
    "Dex5-P1P2-C3Near8mmTeacherResidual-Direct-v0": "Dex5P1P2C3Near8mmTeacherResidualPhysicsEnvCfg",
    "Dex5-P1P2-C3Near5mmTeacherResidual-Direct-v0": "Dex5P1P2C3Near5mmTeacherResidualPhysicsEnvCfg",
    "Dex5-P1P2-C3GeometryTeacherResidual-Direct-v0": "Dex5P1P2C3GeometryTeacherResidualPhysicsEnvCfg",
    "Dex5-P1P2-C3GeometryDenseTeacherResidual-Direct-v0": "Dex5P1P2C3GeometryDenseTeacherResidualPhysicsEnvCfg",
    "Dex5-P1P2-RingOnlyC3TeacherResidual-Direct-v0": (
        "Dex5P1P2RingOnlyC3TeacherResidualPhysicsEnvCfg"
    ),
    "Dex5-P1P2-RingOnly120HzC3TeacherResidual-Direct-v0": (
        "Dex5P1P2RingOnly120HzC3TeacherResidualPhysicsEnvCfg"
    ),
    **{
        f"Dex5-P1P2-RingOnly120HzH{level}-Direct-v0": (
            f"Dex5P1P2RingOnly120HzH{level}PhysicsEnvCfg"
        )
        for level in (1, 2, 3, 6)
    },
    "Dex5-P1P2-TinyTIM120HzH1-Direct-v0": "Dex5P1P2TinyTIM120HzH1PhysicsEnvCfg",
    "Dex5-P1P2-P1617PairedManifold120HzH6-Direct-v0": (
        "Dex5P1P2P1617PairedManifold120HzH6PhysicsEnvCfg"
    ),
    "Dex5-P1P2-P1617PairedManifoldConditioned120HzH6-Direct-v0": (
        "Dex5P1P2P1617PairedManifoldConditioned120HzH6PhysicsEnvCfg"
    ),
    "Dex5-P1P2-P1617SingleRoute539Conditioned120HzH6-Direct-v0": (
        "Dex5P1P2P1617SingleRoute539Conditioned120HzH6PhysicsEnvCfg"
    ),
    **{
        f"Dex5-P1P2-P1617SingleRoute539Conditioned120HzH{level}-Direct-v0": (
            f"Dex5P1P2P1617SingleRoute539Conditioned120HzH{level}PhysicsEnvCfg"
        )
        for level in (1, 2, 3)
    },
    "Dex5-P1P2-P1617SingleRoute539Conditioned120HzH1TerminalWindow-Direct-v0": (
        "Dex5P1P2P1617SingleRoute539Conditioned120HzH1TerminalWindowPhysicsEnvCfg"
    ),
    "Dex5-P1P2-P1617SingleRoute539Conditioned120HzH1TerminalPhase-Direct-v0": (
        "Dex5P1P2P1617SingleRoute539Conditioned120HzH1TerminalPhasePhysicsEnvCfg"
    ),
    "Dex5-P1P2-P1617SingleRoute539Conditioned120HzH1TerminalTarget-Direct-v0": (
        "Dex5P1P2P1617SingleRoute539Conditioned120HzH1TerminalTargetPhysicsEnvCfg"
    ),
    "Dex5-P1P2-P1617SingleRoute539Conditioned120HzH1TerminalDynamics-Direct-v0": (
        "Dex5P1P2P1617SingleRoute539Conditioned120HzH1TerminalDynamicsPhysicsEnvCfg"
    ),
    "Dex5-P1P2-P1617SingleRoute539Conditioned120HzH2TerminalDynamics-Direct-v0": (
        "Dex5P1P2P1617SingleRoute539Conditioned120HzH2TerminalDynamicsPhysicsEnvCfg"
    ),
    "Dex5-P1P2-P1617SingleRoute539Conditioned120HzH3TerminalDynamics-Direct-v0": (
        "Dex5P1P2P1617SingleRoute539Conditioned120HzH3TerminalDynamicsPhysicsEnvCfg"
    ),
    "Dex5-P1P2-P1617SingleRoute539Conditioned120HzH6TerminalDynamics-Direct-v0": (
        "Dex5P1P2P1617SingleRoute539Conditioned120HzH6TerminalDynamicsPhysicsEnvCfg"
    ),
    "Dex5-P1P2-P1617SingleRoute539Conditioned120HzH6TerminalDynamicsPhysicalAudit-Direct-v0": (
        "Dex5P1P2P1617SingleRoute539Conditioned120HzH6TerminalDynamicsPhysicalAuditPhysicsEnvCfg"
    ),
    **{
        f"Dex5-P1P2-P1617Route539LocalArrival120HzH{level}-Direct-v0": (
            f"Dex5P1P2P1617Route539LocalArrival120HzH{level}PhysicsEnvCfg"
        )
        for level in (1, 2, 3, 6)
    },
    "Dex5-P1P2-P1617Route539LocalArrival120HzH6PhysicalAudit-Direct-v0": (
        "Dex5P1P2P1617Route539LocalArrival120HzH6PhysicalAuditPhysicsEnvCfg"
    ),
    "Dex5-P1P2-P1617Route539LocalArrivalNear120HzH1-Direct-v0": (
        "Dex5P1P2P1617Route539LocalArrivalNear120HzH1PhysicsEnvCfg"
    ),
    "Dex5-P1P2-P1617Route539Mixed120HzH1-Direct-v0": (
        "Dex5P1P2P1617Route539Mixed120HzH1PhysicsEnvCfg"
    ),
    **{
        f"Dex5-P1P2-TIMAssist{level}120HzH1-Direct-v0": (
            f"Dex5P1P2TIMAssist{level}120HzH1PhysicsEnvCfg"
        )
        for level in ("005", "0075", "010")
    },
    "Dex5-P1P2-ReverseP2Source-Direct-v0": "Dex5P1P2ReverseP2SourcePhysicsEnvCfg",
    "Dex5-P1P2-ReverseR0Hold-Direct-v0": "Dex5P1P2ReverseR0HoldPhysicsEnvCfg",
    "Dex5-P1P2-ReverseR0ValidatedHold-Direct-v0": "Dex5P1P2ReverseR0ValidatedHoldPhysicsEnvCfg",
    "Dex5-P1P2-ReverseR0ValidatedStabilize-Direct-v0": "Dex5P1P2ReverseR0ValidatedStabilizePhysicsEnvCfg",
    "Dex5-P1P2-ReverseR0ValidatedHoldRestricted-Direct-v0": (
        "Dex5P1P2ReverseR0ValidatedHoldRestrictedPhysicsEnvCfg"
    ),
    "Dex5-P1P2-ReverseR0ValidatedHoldRestricted120Hz-Direct-v0": (
        "Dex5P1P2ReverseR0ValidatedHoldRestricted120HzPhysicsEnvCfg"
    ),
    **{
        f"Dex5-P1P2-ReverseR0ValidatedHoldRestrictedH{hold_level}-Direct-v0": (
            f"Dex5P1P2ReverseR0ValidatedHoldRestrictedH{hold_level}PhysicsEnvCfg"
        )
        for hold_level in range(1, 7)
    },
    "Dex5-P1P2-ReverseR0ValidatedHoldRestrictedH1Continue-Direct-v0": (
        "Dex5P1P2ReverseR0ValidatedHoldRestrictedH1ContinuePhysicsEnvCfg"
    ),
    "Dex5-P1P2-ReverseR0ValidatedHoldFrozenRingH1Continue-Direct-v0": (
        "Dex5P1P2ReverseR0ValidatedHoldFrozenRingH1ContinuePhysicsEnvCfg"
    ),
    "Dex5-P1P2-ReverseR0ValidatedHoldFrozenRingH2-Direct-v0": (
        "Dex5P1P2ReverseR0ValidatedHoldFrozenRingH2PhysicsEnvCfg"
    ),
    "Dex5-P1P2-ReverseR0ValidatedHoldCurrentTargetFrozenRingH2-Direct-v0": (
        "Dex5P1P2ReverseR0ValidatedHoldCurrentTargetFrozenRingH2PhysicsEnvCfg"
    ),
    "Dex5-P1P2-ReverseR0ValidatedHoldRestricted120HzH1-Direct-v0": (
        "Dex5P1P2ReverseR0ValidatedHoldRestricted120HzH1PhysicsEnvCfg"
    ),
    "Dex5-P1P2-ReverseR0ValidatedHoldRestricted120HzH1Continue-Direct-v0": (
        "Dex5P1P2ReverseR0ValidatedHoldRestricted120HzH1ContinuePhysicsEnvCfg"
    ),
    "Dex5-P1P2-ReverseR0ValidatedHoldRestricted120HzH1Bridge3Step-Direct-v0": (
        "Dex5P1P2ReverseR0ValidatedHoldRestricted120HzH1Bridge3StepPhysicsEnvCfg"
    ),
    "Dex5-P1P2-ReverseR0ValidatedHoldRestricted120HzH1Bridge3StepLatched-Direct-v0": (
        "Dex5P1P2ReverseR0ValidatedHoldRestricted120HzH1Bridge3StepLatchedPhysicsEnvCfg"
    ),
    "Dex5-P1P2-ReverseR0ValidatedHoldRestricted120HzH1Bridge3StepLatchedObs-Direct-v0": (
        "Dex5P1P2ReverseR0ValidatedHoldRestricted120HzH1Bridge3StepLatchedObsPhysicsEnvCfg"
    ),
    "Dex5-P1P2-ReverseR0ValidatedHoldRestricted120HzH2-Direct-v0": (
        "Dex5P1P2ReverseR0ValidatedHoldRestricted120HzH2PhysicsEnvCfg"
    ),
    "Dex5-P1P2-StableManifold120HzH6-Direct-v0": (
        "Dex5P1P2StableManifold120HzH6PhysicsEnvCfg"
    ),
    "Dex5-P1P2-ControlledExpansionE1-120HzH6-Direct-v0": (
        "Dex5P1P2ControlledExpansionE1_120HzH6PhysicsEnvCfg"
    ),
    "Dex5-P1P2-ArrivalHoldH1-Direct-v0": "Dex5P1P2ArrivalHoldH1PhysicsEnvCfg",
    "Dex5-P1P2-ArrivalHoldH2-Direct-v0": "Dex5P1P2ArrivalHoldH2PhysicsEnvCfg",
    "Dex5-P1P2-ArrivalHoldH3-Direct-v0": "Dex5P1P2ArrivalHoldH3PhysicsEnvCfg",
    "Dex5-P1P2-ArrivalHoldH4-Direct-v0": "Dex5P1P2ArrivalHoldH4PhysicsEnvCfg",
    "Dex5-P1P2-ArrivalHoldH5-Direct-v0": "Dex5P1P2ArrivalHoldH5PhysicsEnvCfg",
    "Dex5-P1P2-ArrivalHoldH6-Direct-v0": "Dex5P1P2ArrivalHoldH6PhysicsEnvCfg",
    "Dex5-P1P2-ArrivalHoldRestrictedH1-Direct-v0": (
        "Dex5P1P2ArrivalHoldRestrictedH1PhysicsEnvCfg"
    ),
    "Dex5-P1P2-ArrivalHoldRestrictedH2-Direct-v0": (
        "Dex5P1P2ArrivalHoldRestrictedH2PhysicsEnvCfg"
    ),
    "Dex5-P1P2-ArrivalHoldRestrictedH3-Direct-v0": (
        "Dex5P1P2ArrivalHoldRestrictedH3PhysicsEnvCfg"
    ),
    "Dex5-P1P2-ArrivalHoldRestrictedH4-Direct-v0": (
        "Dex5P1P2ArrivalHoldRestrictedH4PhysicsEnvCfg"
    ),
    "Dex5-P1P2-ArrivalHoldRestrictedH5-Direct-v0": (
        "Dex5P1P2ArrivalHoldRestrictedH5PhysicsEnvCfg"
    ),
    "Dex5-P1P2-ArrivalHoldRestrictedH6-Direct-v0": (
        "Dex5P1P2ArrivalHoldRestrictedH6PhysicsEnvCfg"
    ),
    "Dex5-P1P2-ArrivalHoldRestricted120HzH1-Direct-v0": (
        "Dex5P1P2ArrivalHoldRestricted120HzH1PhysicsEnvCfg"
    ),
    "Dex5-P1P2-ArrivalHoldRestricted120HzH2-Direct-v0": (
        "Dex5P1P2ArrivalHoldRestricted120HzH2PhysicsEnvCfg"
    ),
    "Dex5-P1P2-ArrivalHoldRestricted120HzH3-Direct-v0": (
        "Dex5P1P2ArrivalHoldRestricted120HzH3PhysicsEnvCfg"
    ),
    "Dex5-P1P2-ArrivalHoldRestricted120HzH4-Direct-v0": (
        "Dex5P1P2ArrivalHoldRestricted120HzH4PhysicsEnvCfg"
    ),
    "Dex5-P1P2-ArrivalHoldRestricted120HzH5-Direct-v0": (
        "Dex5P1P2ArrivalHoldRestricted120HzH5PhysicsEnvCfg"
    ),
    "Dex5-P1P2-ArrivalHoldRestricted120HzH6-Direct-v0": (
        "Dex5P1P2ArrivalHoldRestricted120HzH6PhysicsEnvCfg"
    ),
    "Dex5-P1P2-FailureFrontier240-120HzH6-Direct-v0": (
        "Dex5P1P2FailureFrontier240_120HzH6PhysicsEnvCfg"
    ),
    "Dex5-P1P2-ReverseR1Hold-Direct-v0": "Dex5P1P2ReverseR1HoldPhysicsEnvCfg",
    "Dex5-P1P2-ReverseR2Hold-Direct-v0": "Dex5P1P2ReverseR2HoldPhysicsEnvCfg",
    "Dex5-P1P2-ReverseR3Hold-Direct-v0": "Dex5P1P2ReverseR3HoldPhysicsEnvCfg",
    "Dex5-P1P2-C3CaptureTeacherResidual-Direct-v0": "Dex5P1P2C3CaptureTeacherResidualPhysicsEnvCfg",
    "Dex5-P1P2-C4-Direct-v0": "Dex5P1P2C4PhysicsEnvCfg",
    "Dex5-P1P2-C5EasyDR-Direct-v0": "Dex5P1P2C5EasyDRPhysicsEnvCfg",
    "Dex5-P1P2-C4-Validation-Direct-v0": "Dex5P1P2C4ValidationPhysicsEnvCfg",
    "Dex5-P1P2-C4-Test-Direct-v0": "Dex5P1P2C4TestPhysicsEnvCfg",
    "Dex5-P1P2-C5EasyDR-Test-Direct-v0": "Dex5P1P2C5EasyDRTestPhysicsEnvCfg",
}

for task_id, config_name in _P1_P2_TASK_CONFIGS.items():
    gym.register(
        id=task_id,
        entry_point=(
            f"{__name__}.p1_p2_physics_env:Dex5P1P2PrimitivePhysicsEnv"
        ),
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                f"{__name__}.p1_p2_physics_env:{config_name}"
            ),
            "skrl_cfg_entry_point": (
                f"{agents.__name__}:skrl_ppo_p1_p2_cfg.yaml"
            ),
        },
    )

gym.register(
    id="Dex5-P1P2-ReverseR0HoldSlow-Direct-v0",
    entry_point=(
        f"{__name__}.p1_p2_physics_env:Dex5P1P2PrimitivePhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.p1_p2_physics_env:Dex5P1P2ReverseR0HoldPhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_ppo_p1_p2_r0_slow_cfg.yaml"
        ),
    },
)

gym.register(
    id="Dex5-P1P2-ReverseR0ValidatedHoldSlow-Direct-v0",
    entry_point=(
        f"{__name__}.p1_p2_physics_env:Dex5P1P2PrimitivePhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.p1_p2_physics_env:"
            "Dex5P1P2ReverseR0ValidatedHoldPhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_ppo_p1_p2_r0_slow_cfg.yaml"
        ),
    },
)

gym.register(
    id="Dex5-P1P2-ReverseR0ValidatedStabilizeSlow-Direct-v0",
    entry_point=(
        f"{__name__}.p1_p2_physics_env:Dex5P1P2PrimitivePhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.p1_p2_physics_env:"
            "Dex5P1P2ReverseR0ValidatedStabilizePhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_ppo_p1_p2_r0_slow_cfg.yaml"
        ),
    },
)

for task_suffix, config_name in (
    (
        "Restricted",
        "Dex5P1P2ReverseR0ValidatedHoldRestrictedPhysicsEnvCfg",
    ),
    (
        "Restricted120Hz",
        "Dex5P1P2ReverseR0ValidatedHoldRestricted120HzPhysicsEnvCfg",
    ),
):
    gym.register(
        id=f"Dex5-P1P2-ReverseR0ValidatedHold{task_suffix}Slow-Direct-v0",
        entry_point=(
            f"{__name__}.p1_p2_physics_env:Dex5P1P2PrimitivePhysicsEnv"
        ),
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                f"{__name__}.p1_p2_physics_env:{config_name}"
            ),
            "skrl_cfg_entry_point": (
                f"{agents.__name__}:skrl_ppo_p1_p2_r0_slow_cfg.yaml"
            ),
        },
    )

gym.register(
    id="Dex5-P1P2-RingOnlyC3TeacherResidualSlow-Direct-v0",
    entry_point=(
        f"{__name__}.p1_p2_physics_env:Dex5P1P2PrimitivePhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.p1_p2_physics_env:"
            "Dex5P1P2RingOnlyC3TeacherResidualPhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_ppo_p1_p2_r0_slow_cfg.yaml"
        ),
    },
)

gym.register(
    id="Dex5-P1P2-RingOnly120HzC3TeacherResidualSlow-Direct-v0",
    entry_point=(
        f"{__name__}.p1_p2_physics_env:Dex5P1P2PrimitivePhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.p1_p2_physics_env:"
            "Dex5P1P2RingOnly120HzC3TeacherResidualPhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_ppo_p1_p2_r0_slow_cfg.yaml"
        ),
    },
)

for hold_level in (1, 2, 3, 6):
    gym.register(
        id=f"Dex5-P1P2-RingOnly120HzH{hold_level}Slow-Direct-v0",
        entry_point=(
            f"{__name__}.p1_p2_physics_env:Dex5P1P2PrimitivePhysicsEnv"
        ),
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                f"{__name__}.p1_p2_physics_env:"
                f"Dex5P1P2RingOnly120HzH{hold_level}PhysicsEnvCfg"
            ),
            "skrl_cfg_entry_point": (
                f"{agents.__name__}:skrl_ppo_p1_p2_r0_slow_cfg.yaml"
            ),
        },
    )

gym.register(
    id="Dex5-P1P2-TinyTIM120HzH1Slow-Direct-v0",
    entry_point=(f"{__name__}.p1_p2_physics_env:Dex5P1P2PrimitivePhysicsEnv"),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.p1_p2_physics_env:Dex5P1P2TinyTIM120HzH1PhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_p1_p2_r0_slow_cfg.yaml",
    },
)

for assist_level in ("005", "0075", "010"):
    gym.register(
        id=f"Dex5-P1P2-TIMAssist{assist_level}120HzH1Slow-Direct-v0",
        entry_point=(f"{__name__}.p1_p2_physics_env:Dex5P1P2PrimitivePhysicsEnv"),
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                f"{__name__}.p1_p2_physics_env:"
                f"Dex5P1P2TIMAssist{assist_level}120HzH1PhysicsEnvCfg"
            ),
            "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_p1_p2_r0_slow_cfg.yaml",
        },
    )

gym.register(
    id="Dex5-P1P2-ReverseR0ValidatedHoldCurrentTargetFrozenRingH2Slow-Direct-v0",
    entry_point=(
        f"{__name__}.p1_p2_physics_env:Dex5P1P2PrimitivePhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.p1_p2_physics_env:"
            "Dex5P1P2ReverseR0ValidatedHoldCurrentTargetFrozenRingH2PhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_ppo_p1_p2_r0_slow_cfg.yaml"
        ),
    },
)

for task_suffix, config_name in (
    (
        "H1",
        "Dex5P1P2ReverseR0ValidatedHoldRestricted120HzH1PhysicsEnvCfg",
    ),
    (
        "H1Continue",
        "Dex5P1P2ReverseR0ValidatedHoldRestricted120HzH1ContinuePhysicsEnvCfg",
    ),
    (
        "H1Bridge3Step",
        "Dex5P1P2ReverseR0ValidatedHoldRestricted120HzH1Bridge3StepPhysicsEnvCfg",
    ),
    (
        "H1Bridge3StepLatched",
        "Dex5P1P2ReverseR0ValidatedHoldRestricted120HzH1Bridge3StepLatchedPhysicsEnvCfg",
    ),
    (
        "H1Bridge3StepLatchedObs",
        "Dex5P1P2ReverseR0ValidatedHoldRestricted120HzH1Bridge3StepLatchedObsPhysicsEnvCfg",
    ),
    (
        "H2",
        "Dex5P1P2ReverseR0ValidatedHoldRestricted120HzH2PhysicsEnvCfg",
    ),
    (
        "StableManifoldH6",
        "Dex5P1P2StableManifold120HzH6PhysicsEnvCfg",
    ),
    (
        "ControlledExpansionE1H6",
        "Dex5P1P2ControlledExpansionE1_120HzH6PhysicsEnvCfg",
    ),
):
    gym.register(
        id=f"Dex5-P1P2-ReverseR0ValidatedHoldRestricted120Hz{task_suffix}Slow-Direct-v0",
        entry_point=(
            f"{__name__}.p1_p2_physics_env:Dex5P1P2PrimitivePhysicsEnv"
        ),
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                f"{__name__}.p1_p2_physics_env:{config_name}"
            ),
            "skrl_cfg_entry_point": (
                f"{agents.__name__}:skrl_ppo_p1_p2_r0_slow_cfg.yaml"
            ),
        },
    )

for hold_level in range(1, 7):
    gym.register(
        id=f"Dex5-P1P2-ReverseR0ValidatedHoldRestrictedH{hold_level}Slow-Direct-v0",
        entry_point=(
            f"{__name__}.p1_p2_physics_env:Dex5P1P2PrimitivePhysicsEnv"
        ),
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                f"{__name__}.p1_p2_physics_env:"
                f"Dex5P1P2ReverseR0ValidatedHoldRestrictedH{hold_level}PhysicsEnvCfg"
            ),
            "skrl_cfg_entry_point": (
                f"{agents.__name__}:skrl_ppo_p1_p2_r0_slow_cfg.yaml"
            ),
        },
    )

gym.register(
    id="Dex5-P1P2-FailureFrontier240-120HzH6Slow-Direct-v0",
    entry_point=(
        f"{__name__}.p1_p2_physics_env:Dex5P1P2PrimitivePhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.p1_p2_physics_env:"
            "Dex5P1P2FailureFrontier240_120HzH6PhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_ppo_p1_p2_r0_slow_cfg.yaml"
        ),
    },
)

gym.register(
    id="Dex5-P1P2-ReverseR0ValidatedHoldRestrictedH1ContinueSlow-Direct-v0",
    entry_point=(
        f"{__name__}.p1_p2_physics_env:Dex5P1P2PrimitivePhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.p1_p2_physics_env:"
            "Dex5P1P2ReverseR0ValidatedHoldRestrictedH1ContinuePhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_ppo_p1_p2_r0_slow_cfg.yaml"
        ),
    },
)

for task_suffix, config_name in (
    (
        "H1Continue",
        "Dex5P1P2ReverseR0ValidatedHoldFrozenRingH1ContinuePhysicsEnvCfg",
    ),
    (
        "H2",
        "Dex5P1P2ReverseR0ValidatedHoldFrozenRingH2PhysicsEnvCfg",
    ),
):
    gym.register(
        id=f"Dex5-P1P2-ReverseR0ValidatedHoldFrozenRing{task_suffix}Slow-Direct-v0",
        entry_point=(
            f"{__name__}.p1_p2_physics_env:Dex5P1P2PrimitivePhysicsEnv"
        ),
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                f"{__name__}.p1_p2_physics_env:{config_name}"
            ),
            "skrl_cfg_entry_point": (
                f"{agents.__name__}:skrl_ppo_p1_p2_r0_slow_cfg.yaml"
            ),
        },
    )

for hold_level in range(1, 7):
    gym.register(
        id=f"Dex5-P1P2-ArrivalHoldH{hold_level}Slow-Direct-v0",
        entry_point=(
            f"{__name__}.p1_p2_physics_env:Dex5P1P2PrimitivePhysicsEnv"
        ),
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                f"{__name__}.p1_p2_physics_env:"
                f"Dex5P1P2ArrivalHoldH{hold_level}PhysicsEnvCfg"
            ),
            "skrl_cfg_entry_point": (
                f"{agents.__name__}:skrl_ppo_p1_p2_r0_slow_cfg.yaml"
            ),
        },
    )

for hold_level in range(1, 7):
    for suffix in ("Restricted", "Restricted120Hz"):
        gym.register(
            id=f"Dex5-P1P2-ArrivalHold{suffix}H{hold_level}Slow-Direct-v0",
            entry_point=(
                f"{__name__}.p1_p2_physics_env:Dex5P1P2PrimitivePhysicsEnv"
            ),
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": (
                    f"{__name__}.p1_p2_physics_env:"
                    f"Dex5P1P2ArrivalHold{suffix}H{hold_level}PhysicsEnvCfg"
                ),
                "skrl_cfg_entry_point": (
                    f"{agents.__name__}:skrl_ppo_p1_p2_r0_slow_cfg.yaml"
                ),
            },
        )

gym.register(
    id="Dex5-V9P4-P5-P1-ContinuousClosureThumbWrapLateTrain-Direct-v0",
    entry_point=(
        f"{__name__}.p5_p1_physics_env:"
        "Dex5V9P4P5P1ContinuousClosureCycleAlignedLateTrainPhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.p5_p1_physics_env:"
            "Dex5V9P4P5P1ContinuousClosureThumbWrapLateTrainPhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml"
        ),
    },
)

gym.register(
    id="Dex5-V9P4-P5-P1-ContinuousClosureThumbUnderLateTrain-Direct-v0",
    entry_point=(
        f"{__name__}.p5_p1_physics_env:"
        "Dex5V9P4P5P1ContinuousClosureCycleAlignedLateTrainPhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.p5_p1_physics_env:"
            "Dex5V9P4P5P1ContinuousClosureThumbUnderLateTrainPhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml"
        ),
    },
)

gym.register(
    id="Dex5-V9P4-P5-P1-ContinuousClosureThumbUnderPhase7LateTrain-Direct-v0",
    entry_point=(
        f"{__name__}.p5_p1_physics_env:"
        "Dex5V9P4P5P1ContinuousClosureCycleAlignedLateTrainPhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.p5_p1_physics_env:"
            "Dex5V9P4P5P1ContinuousClosureThumbUnderPhase7LateTrainPhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml"
        ),
    },
)

gym.register(
    id=(
        "Dex5-V9P4-P5-P1-ContinuousClosureThumbUnder"
        "Phase7To8Train-Direct-v0"
    ),
    entry_point=(
        f"{__name__}.p5_p1_physics_env:"
        "Dex5V9P4P5P1ContinuousClosureCycleAlignedLateTrainPhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.p5_p1_physics_env:"
            "Dex5V9P4P5P1ContinuousClosureThumbUnder"
            "Phase7To8TrainPhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml"
        ),
    },
)

gym.register(
    id="Dex5-V9P4-P5-P1-ContinuousClosureThumbUnderPhase8LateTrain-Direct-v0",
    entry_point=(
        f"{__name__}.p5_p1_physics_env:"
        "Dex5V9P4P5P1ContinuousClosureCycleAlignedLateTrainPhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.p5_p1_physics_env:"
            "Dex5V9P4P5P1ContinuousClosureThumbUnderPhase8LateTrainPhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml"
        ),
    },
)

gym.register(
    id=(
        "Dex5-V9P4-P5-P1-ContinuousClosureThumbUnder"
        "Phase8To9Train-Direct-v0"
    ),
    entry_point=(
        f"{__name__}.p5_p1_physics_env:"
        "Dex5V9P4P5P1ContinuousClosureCycleAlignedLateTrainPhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.p5_p1_physics_env:"
            "Dex5V9P4P5P1ContinuousClosureThumbUnder"
            "Phase8To9TrainPhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml"
        ),
    },
)

gym.register(
    id="Dex5-V9P4-P5-P1-ContinuousClosureThumbWrapPhase9LateTrain-Direct-v0",
    entry_point=(
        f"{__name__}.p5_p1_physics_env:"
        "Dex5V9P4P5P1ContinuousClosureCycleAlignedLateTrainPhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.p5_p1_physics_env:"
            "Dex5V9P4P5P1ContinuousClosureThumbWrapPhase9LateTrainPhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml"
        ),
    },
)

gym.register(
    id="Dex5-V9P4-P5-P1-ContinuousClosureThumbWrapPhase10LateTrain-Direct-v0",
    entry_point=(
        f"{__name__}.p5_p1_physics_env:"
        "Dex5V9P4P5P1ContinuousClosureCycleAlignedLateTrainPhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.p5_p1_physics_env:"
            "Dex5V9P4P5P1ContinuousClosureThumbWrapPhase10LateTrainPhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml"
        ),
    },
)

gym.register(
    id="Dex5-V9P4-P5-P1-ContinuousClosureThumbWrapPhase11LateTrain-Direct-v0",
    entry_point=(
        f"{__name__}.p5_p1_physics_env:"
        "Dex5V9P4P5P1ContinuousClosureCycleAlignedLateTrainPhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.p5_p1_physics_env:"
            "Dex5V9P4P5P1ContinuousClosureThumbWrapPhase11LateTrainPhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml"
        ),
    },
)

gym.register(
    id="Dex5-V9P4-P5-P1-ContinuousClosureThumbWrapPhase12LateTrain-Direct-v0",
    entry_point=(
        f"{__name__}.p5_p1_physics_env:"
        "Dex5V9P4P5P1ContinuousClosureCycleAlignedLateTrainPhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.p5_p1_physics_env:"
            "Dex5V9P4P5P1ContinuousClosureThumbWrapPhase12LateTrainPhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml"
        ),
    },
)

gym.register(
    id="Dex5-V9P4-P5-P1-ContinuousClosureThumbTakeoverPhase13LateTrain-Direct-v0",
    entry_point=(
        f"{__name__}.p5_p1_physics_env:"
        "Dex5V9P4P5P1ContinuousClosureCycleAlignedLateTrainPhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.p5_p1_physics_env:"
            "Dex5V9P4P5P1ContinuousClosureThumbTakeoverPhase13LateTrainPhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml"
        ),
    },
)

gym.register(
    id="Dex5-V9P4-P5-P1-ContinuousClosureThumbTakeoverPhase14LateTrain-Direct-v0",
    entry_point=(
        f"{__name__}.p5_p1_physics_env:"
        "Dex5V9P4P5P1ContinuousClosureCycleAlignedLateTrainPhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.p5_p1_physics_env:"
            "Dex5V9P4P5P1ContinuousClosureThumbTakeoverPhase14LateTrainPhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml"
        ),
    },
)

gym.register(
    id=(
        "Dex5-V9P4-P5-P1-ContinuousClosureThumbTakeover"
        "Phase14Stabilize-Direct-v0"
    ),
    entry_point=(
        f"{__name__}.p5_p1_physics_env:"
        "Dex5V9P4P5P1ContinuousClosureCycleAlignedLateTrainPhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.p5_p1_physics_env:"
            "Dex5V9P4P5P1ContinuousClosureThumbTakeover"
            "Phase14StabilizePhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml"
        ),
    },
)

gym.register(
    id=(
        "Dex5-V9P4-P5-P1-ContinuousClosureThumbTakeover"
        "Phase14EntryCollect-Direct-v0"
    ),
    entry_point=(
        f"{__name__}.p5_p1_physics_env:"
        "Dex5V9P4P5P1ContinuousClosureCycleAlignedLateTrainPhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.p5_p1_physics_env:"
            "Dex5V9P4P5P1ContinuousClosureThumbTakeover"
            "Phase14EntryCollectPhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml"
        ),
    },
)

gym.register(
    id=(
        "Dex5-V9P4-P5-P1-ContinuousClosureMiddleUnload"
        "Phase15Stabilize-Direct-v0"
    ),
    entry_point=(
        f"{__name__}.p5_p1_physics_env:"
        "Dex5V9P4P5P1ContinuousClosureCycleAlignedLateTrainPhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.p5_p1_physics_env:"
            "Dex5V9P4P5P1ContinuousClosureMiddleUnload"
            "Phase15StabilizePhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml"
        ),
    },
)

gym.register(
    id=(
        "Dex5-V9P4-P5-P1-ContinuousClosureMiddleUnload"
        "Phase15EntryTrain-Direct-v0"
    ),
    entry_point=(
        f"{__name__}.p5_p1_physics_env:"
        "Dex5V9P4P5P1ContinuousClosureCycleAlignedLateTrainPhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.p5_p1_physics_env:"
            "Dex5V9P4P5P1ContinuousClosureMiddleUnload"
            "Phase15EntryTrainPhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml"
        ),
    },
)

gym.register(
    id=(
        "Dex5-V9P4-P5-P1-ContinuousClosureMiddleUnload"
        "Phase15EntryStrict-Direct-v0"
    ),
    entry_point=(
        f"{__name__}.p5_p1_physics_env:"
        "Dex5V9P4P5P1ContinuousClosureCycleAlignedLateTrainPhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.p5_p1_physics_env:"
            "Dex5V9P4P5P1ContinuousClosureMiddleUnload"
            "Phase15EntryStrictPhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml"
        ),
    },
)

gym.register(
    id=(
        "Dex5-V9P4-P5-P1-ContinuousClosureMiddleUnload"
        "Phase15EntryMid-Direct-v0"
    ),
    entry_point=(
        f"{__name__}.p5_p1_physics_env:"
        "Dex5V9P4P5P1ContinuousClosureCycleAlignedLateTrainPhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.p5_p1_physics_env:"
            "Dex5V9P4P5P1ContinuousClosureMiddleUnload"
            "Phase15EntryMidPhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml"
        ),
    },
)

gym.register(
    id=(
        "Dex5-V9P4-P5-P1-ContinuousClosureTakeover"
        "Phase14To15NoReset-Direct-v0"
    ),
    entry_point=(
        f"{__name__}.p5_p1_physics_env:"
        "Dex5V9P4P5P1ContinuousClosureCycleAlignedLateTrainPhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.p5_p1_physics_env:"
            "Dex5V9P4P5P1ContinuousClosureTakeover"
            "Phase14To15NoResetPhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml"
        ),
    },
)

gym.register(
    id=(
        "Dex5-V9P4-P5-P1-ContinuousClosureTakeover"
        "Phase14To15Margin75-Direct-v0"
    ),
    entry_point=(
        f"{__name__}.p5_p1_physics_env:"
        "Dex5V9P4P5P1ContinuousClosureCycleAlignedLateTrainPhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.p5_p1_physics_env:"
            "Dex5V9P4P5P1ContinuousClosureTakeover"
            "Phase14To15Margin75PhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml"
        ),
    },
)

gym.register(
    id=(
        "Dex5-V9P4-P5-P1-ContinuousClosureTakeover"
        "Phase14To15TeacherPrefill-Direct-v0"
    ),
    entry_point=(
        f"{__name__}.p5_p1_physics_env:"
        "Dex5V9P4P5P1ContinuousClosureCycleAlignedLateTrainPhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.p5_p1_physics_env:"
            "Dex5V9P4P5P1ContinuousClosureTakeover"
            "Phase14To15TeacherPrefillPhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml"
        ),
    },
)

gym.register(
    id=(
        "Dex5-V9P4-P5-P1-ContinuousClosureTakeover"
        "Phase15BCAuthority-Direct-v0"
    ),
    entry_point=(
        f"{__name__}.p5_p1_physics_env:"
        "Dex5V9P4P5P1ContinuousClosureCycleAlignedLateTrainPhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.p5_p1_physics_env:"
            "Dex5V9P4P5P1ContinuousClosureTakeover"
            "Phase15BCAuthorityPhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml"
        ),
    },
)

gym.register(
    id=(
        "Dex5-V9P4-P5-P1-ContinuousClosure"
        "Phase16MiddleWithdrawal-Direct-v0"
    ),
    entry_point=(
        f"{__name__}.p5_p1_physics_env:"
        "Dex5V9P4P5P1ContinuousClosureCycleAlignedLateTrainPhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.p5_p1_physics_env:"
            "Dex5V9P4P5P1ContinuousClosure"
            "Phase16MiddleWithdrawalPhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml"
        ),
    },
)

gym.register(
    id=(
        "Dex5-V9P4-P5-P1-ContinuousClosure"
        "Phase16MiddleWithdrawalApproach-Direct-v0"
    ),
    entry_point=(
        f"{__name__}.p5_p1_physics_env:"
        "Dex5V9P4P5P1ContinuousClosureCycleAlignedLateTrainPhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.p5_p1_physics_env:"
            "Dex5V9P4P5P1ContinuousClosure"
            "Phase16MiddleWithdrawalApproachPhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml"
        ),
    },
)

gym.register(
    id=(
        "Dex5-V9P4-P5-P1-ContinuousClosure"
        "Phase16ATo17NoReset-Direct-v0"
    ),
    entry_point=(
        f"{__name__}.p5_p1_physics_env:"
        "Dex5V9P4P5P1ContinuousClosureCycleAlignedLateTrainPhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.p5_p1_physics_env:"
            "Dex5V9P4P5P1ContinuousClosure"
            "Phase16ATo17NoResetPhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml"
        ),
    },
)

gym.register(
    id=(
        "Dex5-V9P4-P5-P1-ContinuousClosure"
        "Phase18MiddleWithdrawal-Direct-v0"
    ),
    entry_point=(
        f"{__name__}.p5_p1_physics_env:"
        "Dex5V9P4P5P1ContinuousClosureCycleAlignedLateTrainPhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.p5_p1_physics_env:"
            "Dex5V9P4P5P1ContinuousClosure"
            "Phase18MiddleWithdrawalPhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml"
        ),
    },
)

gym.register(
    id="Dex5-V9P4-P5-P1-DualEntryThumbIndexPinch-Direct-v0",
    entry_point=(
        f"{__name__}.p5_p1_physics_env:"
        "Dex5V9P4P5P1ContinuousClosureCycleAlignedLateTrainPhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.p5_p1_physics_env:"
            "Dex5V9P4P5P1DualEntryThumbIndexPinchPhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml"
        ),
    },
)

gym.register(
    id=(
        "Dex5-V9P4-P5-P1-ContinuousClosure"
        "Phase17TeacherPrefill-Direct-v0"
    ),
    entry_point=(
        f"{__name__}.p5_p1_physics_env:"
        "Dex5V9P4P5P1ContinuousClosureCycleAlignedLateTrainPhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.p5_p1_physics_env:"
            "Dex5V9P4P5P1ContinuousClosure"
            "Phase17TeacherPrefillPhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml"
        ),
    },
)

gym.register(
    id=(
        "Dex5-V9P4-P5-P1-ContinuousClosure"
        "Phase17StrictTeacherResidual-Direct-v0"
    ),
    entry_point=(
        f"{__name__}.p5_p1_physics_env:"
        "Dex5V9P4P5P1ContinuousClosureCycleAlignedLateTrainPhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.p5_p1_physics_env:"
            "Dex5V9P4P5P1ContinuousClosure"
            "Phase17StrictTeacherResidualPhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml"
        ),
    },
)

gym.register(
    id=(
        "Dex5-V9P4-P5-P1-ContinuousClosure"
        "Phase17TeacherResidualMargin12-Direct-v0"
    ),
    entry_point=(
        f"{__name__}.p5_p1_physics_env:"
        "Dex5V9P4P5P1ContinuousClosureCycleAlignedLateTrainPhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.p5_p1_physics_env:"
            "Dex5V9P4P5P1ContinuousClosure"
            "Phase17TeacherResidualMargin12PhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml"
        ),
    },
)

gym.register(
    id=(
        "Dex5-V9P4-P5-P1-ContinuousClosure"
        "Phase17StandaloneMargin12-Direct-v0"
    ),
    entry_point=(
        f"{__name__}.p5_p1_physics_env:"
        "Dex5V9P4P5P1ContinuousClosureCycleAlignedLateTrainPhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.p5_p1_physics_env:"
            "Dex5V9P4P5P1ContinuousClosure"
            "Phase17StandaloneMargin12PhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml"
        ),
    },
)

gym.register(
    id=(
        "Dex5-V9P4-P5-P1-ContinuousClosure"
        "Phase17MeanTeacherMargin12-Direct-v0"
    ),
    entry_point=(
        f"{__name__}.p5_p1_physics_env:"
        "Dex5V9P4P5P1ContinuousClosureCycleAlignedLateTrainPhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.p5_p1_physics_env:"
            "Dex5V9P4P5P1ContinuousClosure"
            "Phase17MeanTeacherMargin12PhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml"
        ),
    },
)

gym.register(
    id=(
        "Dex5-V9P4-P5-P1-ContinuousClosure"
        "Phase17StandaloneMargin11-Direct-v0"
    ),
    entry_point=(
        f"{__name__}.p5_p1_physics_env:"
        "Dex5V9P4P5P1ContinuousClosureCycleAlignedLateTrainPhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.p5_p1_physics_env:"
            "Dex5V9P4P5P1ContinuousClosure"
            "Phase17StandaloneMargin11PhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml"
        ),
    },
)

gym.register(
    id=(
        "Dex5-V9P4-P5-P1-ContinuousClosure"
        "Phase17TeacherResidualMargin11-Direct-v0"
    ),
    entry_point=(
        f"{__name__}.p5_p1_physics_env:"
        "Dex5V9P4P5P1ContinuousClosureCycleAlignedLateTrainPhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.p5_p1_physics_env:"
            "Dex5V9P4P5P1ContinuousClosure"
            "Phase17TeacherResidualMargin11PhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml"
        ),
    },
)

gym.register(
    id=(
        "Dex5-V9P4-P5-P1-ContinuousClosure"
        "Phase17TeacherResidualMargin115-Direct-v0"
    ),
    entry_point=(
        f"{__name__}.p5_p1_physics_env:"
        "Dex5V9P4P5P1ContinuousClosureCycleAlignedLateTrainPhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.p5_p1_physics_env:"
            "Dex5V9P4P5P1ContinuousClosure"
            "Phase17TeacherResidualMargin115PhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml"
        ),
    },
)

gym.register(
    id=(
        "Dex5-V9P4-P5-P1-ContinuousClosure"
        "Phase17TeacherResidualMargin14-Direct-v0"
    ),
    entry_point=(
        f"{__name__}.p5_p1_physics_env:"
        "Dex5V9P4P5P1ContinuousClosureCycleAlignedLateTrainPhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.p5_p1_physics_env:"
            "Dex5V9P4P5P1ContinuousClosure"
            "Phase17TeacherResidualMargin14PhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml"
        ),
    },
)

gym.register(
    id=(
        "Dex5-V9P4-P5-P1-ContinuousClosure"
        "Phase17TeacherResidualMargin13-Direct-v0"
    ),
    entry_point=(
        f"{__name__}.p5_p1_physics_env:"
        "Dex5V9P4P5P1ContinuousClosureCycleAlignedLateTrainPhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.p5_p1_physics_env:"
            "Dex5V9P4P5P1ContinuousClosure"
            "Phase17TeacherResidualMargin13PhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml"
        ),
    },
)

gym.register(
    id="Dex5-V9P4-P5-P1-ContinuousClosureRingUnloadPhase15LateTrain-Direct-v0",
    entry_point=(
        f"{__name__}.p5_p1_physics_env:"
        "Dex5V9P4P5P1ContinuousClosureCycleAlignedLateTrainPhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.p5_p1_physics_env:"
            "Dex5V9P4P5P1ContinuousClosureRingUnloadPhase15LateTrainPhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml"
        ),
    },
)

gym.register(
    id="Dex5-V9P4-P5-P1-ContinuousClosureRingUnloadPhase16LateTrain-Direct-v0",
    entry_point=(
        f"{__name__}.p5_p1_physics_env:"
        "Dex5V9P4P5P1ContinuousClosureCycleAlignedLateTrainPhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.p5_p1_physics_env:"
            "Dex5V9P4P5P1ContinuousClosureRingUnloadPhase16LateTrainPhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml"
        ),
    },
)

gym.register(
    id="Dex5-V9P4-P5-P1-ContinuousClosureP1FinalPhase17LateTrain-Direct-v0",
    entry_point=(
        f"{__name__}.p5_p1_physics_env:"
        "Dex5V9P4P5P1ContinuousClosureCycleAlignedLateTrainPhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.p5_p1_physics_env:"
            "Dex5V9P4P5P1ContinuousClosureP1FinalPhase17LateTrainPhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml"
        ),
    },
)

gym.register(
    id=(
        "Dex5-V9P4-P5-P1-ContinuousClosureCycleAlignedDenseTorqueLateTrain-"
        "Direct-v0"
    ),
    entry_point=(
        f"{__name__}.p5_p1_physics_env:"
        "Dex5V9P4P5P1ContinuousClosureCycleAlignedLateTrainPhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.p5_p1_physics_env:"
            "Dex5V9P4P5P1ContinuousClosureCycleAlignedDenseTorqueLateTrainPhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml"
        ),
    },
)

gym.register(
    id=(
        "Dex5-V9P4-P5-P1-ContinuousClosureCycleAlignedDenseTorqueBrakeLateTrain-"
        "Direct-v0"
    ),
    entry_point=(
        f"{__name__}.p5_p1_physics_env:"
        "Dex5V9P4P5P1ContinuousClosureCycleAlignedLateTrainPhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.p5_p1_physics_env:"
            "Dex5V9P4P5P1ContinuousClosureCycleAlignedDenseTorqueBrakeLateTrainPhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml"
        ),
    },
)

gym.register(
    id=(
        "Dex5-V9P4-P5-P1-ContinuousClosureCycleAlignedDenseTorque"
        "BrakeThumbLateTrain-Direct-v0"
    ),
    entry_point=(
        f"{__name__}.p5_p1_physics_env:"
        "Dex5V9P4P5P1ContinuousClosureCycleAlignedLateTrainPhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.p5_p1_physics_env:"
            "Dex5V9P4P5P1ContinuousClosureCycleAlignedDenseTorque"
            "BrakeThumbLateTrainPhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml"
        ),
    },
)

gym.register(
    id="Dex5-IndexContact-Direct-v0",
    entry_point=f"{__name__}.recapture_p3_physics_env:Dex5IndexContactPhysicsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.recapture_p3_physics_env:Dex5IndexContactPhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml",
    },
)

gym.register(
    id="Dex5-EarlierPreRingIndexContact-Direct-v0",
    entry_point=(
        f"{__name__}.recapture_p3_physics_env:Dex5EarlierPreRingIndexContactPhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.recapture_p3_physics_env:Dex5EarlierPreRingIndexContactPhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml",
    },
)

gym.register(
    id="Dex5-EarlierToP3Entry-Direct-v0",
    entry_point=f"{__name__}.recapture_p3_physics_env:Dex5EarlierToP3EntryPhysicsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.recapture_p3_physics_env:Dex5EarlierToP3EntryPhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml",
    },
)

gym.register(
    id="Dex5-EarlierToP3EntryTeacher-Direct-v0",
    entry_point=(
        f"{__name__}.recapture_p3_physics_env:Dex5EarlierToP3EntryTeacherPhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.recapture_p3_physics_env:"
            "Dex5EarlierToP3EntryTeacherPhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml",
    },
)

gym.register(
    id="Dex5-IndexRecapture-Direct-v0",
    entry_point=f"{__name__}.recapture_p3_physics_env:Dex5IndexRecapturePhysicsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.recapture_p3_physics_env:Dex5IndexRecapturePhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml",
    },
)

gym.register(
    id="Dex5-RecaptureP3-Direct-v0",
    entry_point=f"{__name__}.recapture_p3_physics_env:Dex5RecaptureP3PhysicsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.recapture_p3_physics_env:Dex5RecaptureP3PhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml",
    },
)

gym.register(
    id="Dex5-RingTouch-Direct-v0",
    entry_point=f"{__name__}.recapture_p3_physics_env:Dex5RingTouchPhysicsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.recapture_p3_physics_env:Dex5RingTouchPhysicsEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml",
    },
)

gym.register(
    id="Dex5-P3Entry-Direct-v0",
    entry_point=f"{__name__}.recapture_p3_physics_env:Dex5P3EntryPhysicsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.recapture_p3_physics_env:Dex5P3EntryPhysicsEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml",
    },
)

gym.register(
    id="Dex5-P3Hold-Direct-v0",
    entry_point=f"{__name__}.recapture_p3_physics_env:Dex5P3HoldPhysicsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.recapture_p3_physics_env:Dex5P3HoldPhysicsEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml",
    },
)

gym.register(
    id="Dex5-V9P3-DynamicHoldTeacher-Direct-v0",
    entry_point=f"{__name__}.recapture_p3_physics_env:Dex5V9P3DynamicHoldTeacherPhysicsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.recapture_p3_physics_env:Dex5V9P3DynamicHoldTeacherPhysicsEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml",
    },
)

gym.register(
    id="Dex5-V9P3-StrictHold-Direct-v0",
    entry_point=f"{__name__}.recapture_p3_physics_env:Dex5V9P3StrictHoldPhysicsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.recapture_p3_physics_env:Dex5V9P3StrictHoldPhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml",
    },
)

gym.register(
    id="Dex5-V9P4-P5-Direct-v0",
    entry_point=f"{__name__}.p4_p5_physics_env:Dex5V9P4P5PhysicsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.p4_p5_physics_env:Dex5V9P4P5PhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml",
    },
)

for task_id, cfg_name in (
    ("Dex5-V9P4-Hold-Direct-v0", "Dex5V9P4HoldPhysicsEnvCfg"),
    ("Dex5-V9P4-P5-MiddleClear-Direct-v0", "Dex5V9P4P5MiddleClearPhysicsEnvCfg"),
    (
        "Dex5-V9P4-P5-MiddleClearPinch-Direct-v0",
        "Dex5V9P4P5MiddleClearPinchPhysicsEnvCfg",
    ),
    (
        "Dex5-V9P4-P5-MiddleClearTeacher-Direct-v0",
        "Dex5V9P4P5MiddleClearTeacherPhysicsEnvCfg",
    ),
    (
        "Dex5-V9P4-P5-MiddleClearBlend75-Direct-v0",
        "Dex5V9P4P5MiddleClearBlend75PhysicsEnvCfg",
    ),
    (
        "Dex5-V9P4-P5-MiddleClearBlend50-Direct-v0",
        "Dex5V9P4P5MiddleClearBlend50PhysicsEnvCfg",
    ),
    (
        "Dex5-V9P4-P5-MiddleClearBlend25-Direct-v0",
        "Dex5V9P4P5MiddleClearBlend25PhysicsEnvCfg",
    ),
    (
        "Dex5-V9P4-P5-MiddleClearBlend0-Direct-v0",
        "Dex5V9P4P5MiddleClearBlend0PhysicsEnvCfg",
    ),
    ("Dex5-V9P4-P5-RingClear-Direct-v0", "Dex5V9P4P5RingClearPhysicsEnvCfg"),
    ("Dex5-V9P4-P5-JointClear-Direct-v0", "Dex5V9P4P5JointClearPhysicsEnvCfg"),
    ("Dex5-V9P4-P5-Capture-Direct-v0", "Dex5V9P4P5CapturePhysicsEnvCfg"),
    (
        "Dex5-V9P4-P5-CaptureBlend75-Direct-v0",
        "Dex5V9P4P5CaptureBlend75PhysicsEnvCfg",
    ),
    (
        "Dex5-V9P4-P5-CaptureBlend50-Direct-v0",
        "Dex5V9P4P5CaptureBlend50PhysicsEnvCfg",
    ),
    (
        "Dex5-V9P4-P5-CaptureBlend25-Direct-v0",
        "Dex5V9P4P5CaptureBlend25PhysicsEnvCfg",
    ),
    (
        "Dex5-V9P4-P5-CaptureBlend0-Direct-v0",
        "Dex5V9P4P5CaptureBlend0PhysicsEnvCfg",
    ),
    ("Dex5-V9P4-P5-Unified-Direct-v0", "Dex5V9P4P5UnifiedPhysicsEnvCfg"),
    (
        "Dex5-V9P4-P5-UnifiedEasyDR-Direct-v0",
        "Dex5V9P4P5UnifiedEasyDRPhysicsEnvCfg",
    ),
    (
        "Dex5-V9P4-P5-UnifiedMediumDR-Direct-v0",
        "Dex5V9P4P5UnifiedMediumDRPhysicsEnvCfg",
    ),
    (
        "Dex5-V9P4-P5-UnifiedHardDR-Direct-v0",
        "Dex5V9P4P5UnifiedHardDRPhysicsEnvCfg",
    ),
    (
        "Dex5-V9P4-P5-UpstreamMiddleClear-Direct-v0",
        "Dex5V9P4P5UpstreamMiddleClearPhysicsEnvCfg",
    ),
    (
        "Dex5-V9P4-P5-CurrentP4MiddleClear-Direct-v0",
        "Dex5V9P4P5CurrentP4MiddleClearPhysicsEnvCfg",
    ),
    (
        "Dex5-V9P4-P5-CurrentP4Capture-Direct-v0",
        "Dex5V9P4P5CurrentP4CapturePhysicsEnvCfg",
    ),
    (
        "Dex5-V9P4-P5-UpstreamCapture-Direct-v0",
        "Dex5V9P4P5UpstreamCapturePhysicsEnvCfg",
    ),
    (
        "Dex5-V9P4-P5-UpstreamCaptureEasyDR-Direct-v0",
        "Dex5V9P4P5UpstreamCaptureEasyDRPhysicsEnvCfg",
    ),
    (
        "Dex5-V9P4-P5-UpstreamLearnedCapture-Direct-v0",
        "Dex5V9P4P5UpstreamLearnedCapturePhysicsEnvCfg",
    ),
    (
        "Dex5-V9P4-P5-UpstreamLearnedCaptureHold200-Direct-v0",
        "Dex5V9P4P5UpstreamLearnedCaptureHold200PhysicsEnvCfg",
    ),
    (
        "Dex5-V9P4-P5-UpstreamLearnedCaptureEasyDR-Direct-v0",
        "Dex5V9P4P5UpstreamLearnedCaptureEasyDRPhysicsEnvCfg",
    ),
    (
        "Dex5-V9P4-P5-UpstreamLearnedCaptureMediumDR-Direct-v0",
        "Dex5V9P4P5UpstreamLearnedCaptureMediumDRPhysicsEnvCfg",
    ),
    (
        "Dex5-V9P4-P5-UpstreamLearnedCaptureEasyDRHold200-Direct-v0",
        "Dex5V9P4P5UpstreamLearnedCaptureEasyDRHold200PhysicsEnvCfg",
    ),
    (
        "Dex5-V9P4-P5-UpstreamLearnedCaptureMediumDRHold200-Direct-v0",
        "Dex5V9P4P5UpstreamLearnedCaptureMediumDRHold200PhysicsEnvCfg",
    ),
):
    gym.register(
        id=task_id,
        entry_point=f"{__name__}.p4_p5_physics_env:Dex5V9P4P5PhysicsEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                f"{__name__}.p4_p5_physics_env:{cfg_name}"
            ),
            "skrl_cfg_entry_point": (
                f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml"
            ),
        },
    )

for task_id, cfg_name in (
    ("Dex5-V9P5-Hold-Direct-v0", "Dex5V9P5HoldPhysicsEnvCfg"),
    (
        "Dex5-V9P5-P1-MiddleApproach-Direct-v0",
        "Dex5V9P5P1MiddleApproachPhysicsEnvCfg",
    ),
    (
        "Dex5-V9P5-P1-MiddleContact-Direct-v0",
        "Dex5V9P5P1MiddleContactPhysicsEnvCfg",
    ),
    ("Dex5-V9P5-P1-Triad-Direct-v0", "Dex5V9P5P1TriadPhysicsEnvCfg"),
    ("Dex5-V9P5-P1-Strict-Direct-v0", "Dex5V9P5P1StrictPhysicsEnvCfg"),
    (
        "Dex5-V9P5-P1-StrictHold200-Direct-v0",
        "Dex5V9P5P1StrictHold200PhysicsEnvCfg",
    ),
    (
        "Dex5-V9P5-P1-StrictHold200MediumDR-Direct-v0",
        "Dex5V9P5P1StrictHold200MediumDRPhysicsEnvCfg",
    ),
    (
        "Dex5-V9P5-P1-ContinuousApproach-Direct-v0",
        "Dex5V9P5P1ContinuousApproachPhysicsEnvCfg",
    ),
    (
        "Dex5-V9P5-P1-ContinuousContact-Direct-v0",
        "Dex5V9P5P1ContinuousContactPhysicsEnvCfg",
    ),
    (
        "Dex5-V9P5-P1-ContinuousTriad-Direct-v0",
        "Dex5V9P5P1ContinuousTriadPhysicsEnvCfg",
    ),
    (
        "Dex5-V9P5-P1-ContinuousStrict-Direct-v0",
        "Dex5V9P5P1ContinuousStrictPhysicsEnvCfg",
    ),
    (
        "Dex5-V9P5-P1-ContinuousStrictHold200-Direct-v0",
        "Dex5V9P5P1ContinuousStrictHold200PhysicsEnvCfg",
    ),
    ("Dex5-V9P5-P1-Unified-Direct-v0", "Dex5V9P5P1UnifiedPhysicsEnvCfg"),
):
    gym.register(
        id=task_id,
        entry_point=f"{__name__}.p5_p1_physics_env:Dex5V9P5P1PhysicsEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                f"{__name__}.p5_p1_physics_env:{cfg_name}"
            ),
            "skrl_cfg_entry_point": (
                f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml"
            ),
        },
    )

gym.register(
    id=(
        "Dex5-V9P4-P5-P1-ContinuousClosureCycleAlignedLateTrain-"
        "Direct-v0"
    ),
    entry_point=(
        f"{__name__}.p5_p1_physics_env:"
        "Dex5V9P4P5P1ContinuousClosureCycleAlignedLateTrainPhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.p5_p1_physics_env:"
            "Dex5V9P4P5P1ContinuousClosureCycleAlignedLateTrainPhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml"
        ),
    },
)

gym.register(
    id="Dex5-V9P4-P5-P1-NoReset-Direct-v0",
    entry_point=(
        f"{__name__}.p5_p1_physics_env:"
        "Dex5V9P4P5P1NoResetPhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.p5_p1_physics_env:"
            "Dex5V9P4P5P1NoResetPhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml"
        ),
    },
)

gym.register(
    id="Dex5-V9P4-P5-P1-ContinuousTriadTrain-Direct-v0",
    entry_point=(
        f"{__name__}.p5_p1_physics_env:"
        "Dex5V9P4P5P1ContinuousTriadTrainPhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.p5_p1_physics_env:"
            "Dex5V9P4P5P1ContinuousTriadTrainPhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml"
        ),
    },
)

gym.register(
    id="Dex5-V9P4-P5-P1-ContinuousContactTriadTrain-Direct-v0",
    entry_point=(
        f"{__name__}.p5_p1_physics_env:"
        "Dex5V9P4P5P1ContinuousTriadTrainPhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.p5_p1_physics_env:"
            "Dex5V9P4P5P1ContinuousContactTriadTrainPhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": (
            f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml"
        ),
    },
)

for task_id, cfg_name in (
    (
        "Dex5-V9P4-P5-P1-ContinuousContactTriadBlend50Train-Direct-v0",
        "Dex5V9P4P5P1ContinuousContactTriadBlend50TrainPhysicsEnvCfg",
    ),
    (
        "Dex5-V9P4-P5-P1-ContinuousContactTriadBlend75Train-Direct-v0",
        "Dex5V9P4P5P1ContinuousContactTriadBlend75TrainPhysicsEnvCfg",
    ),
    (
        "Dex5-V9P4-P5-P1-ContinuousContactTriadCanonicalTrain-Direct-v0",
        "Dex5V9P4P5P1ContinuousContactTriadCanonicalTrainPhysicsEnvCfg",
    ),
    (
        "Dex5-V9P4-P5-P1-ContinuousContactTriadBlend50Slow-Direct-v0",
        "Dex5V9P4P5P1ContinuousContactTriadBlend50SlowPhysicsEnvCfg",
    ),
    (
        "Dex5-V9P4-P5-P1-ContinuousContactTriadBlend50Gentle-Direct-v0",
        "Dex5V9P4P5P1ContinuousContactTriadBlend50GentlePhysicsEnvCfg",
    ),
    (
        "Dex5-V9P4-P5-P1-ContinuousContactTriadBlend50Fast-Direct-v0",
        "Dex5V9P4P5P1ContinuousContactTriadBlend50FastPhysicsEnvCfg",
    ),
    (
        "Dex5-V9P4-P5-P1-ContinuousContactTriadBlend50Residual-Direct-v0",
        "Dex5V9P4P5P1ContinuousContactTriadBlend50ResidualPhysicsEnvCfg",
    ),
    (
        "Dex5-V9P4-P5-P1-ContinuousContactTriadBlend75Gentle-Direct-v0",
        "Dex5V9P4P5P1ContinuousContactTriadBlend75GentlePhysicsEnvCfg",
    ),
    (
        "Dex5-V9P4-P5-P1-ContinuousContactTriadCanonicalGentle-Direct-v0",
        "Dex5V9P4P5P1ContinuousContactTriadCanonicalGentlePhysicsEnvCfg",
    ),
    (
        "Dex5-V9P4-P5-P1-ContinuousClosureCurriculum-Direct-v0",
        "Dex5V9P4P5P1ContinuousClosureCurriculumPhysicsEnvCfg",
    ),
    (
        "Dex5-V9P4-P5-P1-ContinuousClosureHold100-Direct-v0",
        "Dex5V9P4P5P1ContinuousClosureHold100PhysicsEnvCfg",
    ),
    (
        "Dex5-V9P4-P5-P1-ContinuousClosureFine-Direct-v0",
        "Dex5V9P4P5P1ContinuousClosureFinePhysicsEnvCfg",
    ),
    (
        "Dex5-V9P4-P5-P1-ContinuousClosureFineSlow-Direct-v0",
        "Dex5V9P4P5P1ContinuousClosureFineSlowPhysicsEnvCfg",
    ),
    (
        "Dex5-V9P4-P5-P1-ContinuousClosureProbe-Direct-v0",
        "Dex5V9P4P5P1ContinuousClosureProbePhysicsEnvCfg",
    ),
    (
        "Dex5-V9P4-P5-P1-ContinuousClosureProbeSlow-Direct-v0",
        "Dex5V9P4P5P1ContinuousClosureProbeSlowPhysicsEnvCfg",
    ),
    (
        "Dex5-V9P4-P5-P1-ContinuousClosureCycleAlignedProbe-Direct-v0",
        "Dex5V9P4P5P1ContinuousClosureCycleAlignedProbePhysicsEnvCfg",
    ),
    (
        "Dex5-V9P4-P5-P1-ContinuousClosureCycleAlignedApproachM075-Direct-v0",
        "Dex5V9P4P5P1ContinuousClosureCycleAlignedApproachM075PhysicsEnvCfg",
    ),
    (
        "Dex5-V9P4-P5-P1-ContinuousClosureCycleAlignedApproachM100-Direct-v0",
        "Dex5V9P4P5P1ContinuousClosureCycleAlignedApproachM100PhysicsEnvCfg",
    ),
    (
        "Dex5-V9P4-P5-P1-ContinuousClosureCycleAlignedContactScale050-Direct-v0",
        "Dex5V9P4P5P1ContinuousClosureCycleAlignedContactScale050PhysicsEnvCfg",
    ),
    (
        "Dex5-V9P4-P5-P1-ContinuousClosureCycleAlignedContactScale075-Direct-v0",
        "Dex5V9P4P5P1ContinuousClosureCycleAlignedContactScale075PhysicsEnvCfg",
    ),
    (
        "Dex5-V9P4-P5-P1-ContinuousClosureCycleAlignedContact050LateTrain-Direct-v0",
        "Dex5V9P4P5P1ContinuousClosureCycleAlignedContact050LateTrainPhysicsEnvCfg",
    ),
    (
        "Dex5-V9P4-P5-P1-ContinuousClosureCycleAlignedPenPoseTrain-Direct-v0",
        "Dex5V9P4P5P1ContinuousClosureCycleAlignedPenPoseTrainPhysicsEnvCfg",
    ),
    (
        "Dex5-V9P4-P5-P1-ContinuousClosureCycleAlignedDensePenPoseTrain-Direct-v0",
        "Dex5V9P4P5P1ContinuousClosureCycleAlignedDensePenPoseTrainPhysicsEnvCfg",
    ),
    (
        "Dex5-V9P4-P5-P1-ContinuousClosureCycleAlignedDenseTorqueTrain-Direct-v0",
        "Dex5V9P4P5P1ContinuousClosureCycleAlignedDenseTorqueTrainPhysicsEnvCfg",
    ),
    (
        "Dex5-V9P4-P5-P1-ContinuousClosureCycleAlignedDenseTorqueBrakeTrain-Direct-v0",
        "Dex5V9P4P5P1ContinuousClosureCycleAlignedDenseTorqueBrakeTrainPhysicsEnvCfg",
    ),
    (
        "Dex5-V9P4-P5-P1-ContinuousClosureCycleAlignedDenseTorqueBrakeThumbTrain-Direct-v0",
        "Dex5V9P4P5P1ContinuousClosureCycleAlignedDenseTorqueBrakeThumbTrainPhysicsEnvCfg",
    ),
    (
        "Dex5-V9P4-P5-P1-ContinuousClosureThumbWrapTrain-Direct-v0",
        "Dex5V9P4P5P1ContinuousClosureThumbWrapTrainPhysicsEnvCfg",
    ),
    (
        "Dex5-V9P4-P5-P1-ContinuousClosureThumbWrapNoResetTeacher-Direct-v0",
        "Dex5V9P4P5P1ContinuousClosureThumbWrapNoResetTeacherPhysicsEnvCfg",
    ),
    (
        "Dex5-V9P4-P5-P1-ContinuousClosureCycleAlignedTrain-Direct-v0",
        "Dex5V9P4P5P1ContinuousClosureCycleAlignedTrainPhysicsEnvCfg",
    ),
    (
        "Dex5-V9P4-P5-P1-ContinuousClosureCycleAlignedTrainScale005-Direct-v0",
        "Dex5V9P4P5P1ContinuousClosureCycleAlignedTrainScale005PhysicsEnvCfg",
    ),
    (
        "Dex5-V9P4-P5-P1-ContinuousClosureCycleAlignedTrainScale015-Direct-v0",
        "Dex5V9P4P5P1ContinuousClosureCycleAlignedTrainScale015PhysicsEnvCfg",
    ),
    (
        "Dex5-V9P4-P5-P1-ContinuousClosureCycleAlignedEarlyTrain-Direct-v0",
        "Dex5V9P4P5P1ContinuousClosureCycleAlignedEarlyTrainPhysicsEnvCfg",
    ),
    (
        "Dex5-V9P4-P5-P1-ContinuousClosureCycleAlignedContactTrain-Direct-v0",
        "Dex5V9P4P5P1ContinuousClosureCycleAlignedContactTrainPhysicsEnvCfg",
    ),
    (
        "Dex5-V9P4-P5-P1-ContinuousClosureCycleAlignedApproachTrain-Direct-v0",
        "Dex5V9P4P5P1ContinuousClosureCycleAlignedApproachTrainPhysicsEnvCfg",
    ),
    (
        "Dex5-V9P4-P5-P1-ContinuousClosureCycleAlignedApproachScale002-Direct-v0",
        "Dex5V9P4P5P1ContinuousClosureCycleAlignedApproachScale002PhysicsEnvCfg",
    ),
    (
        "Dex5-V9P4-P5-P1-ContinuousClosureCycleAlignedApproachScale0035-Direct-v0",
        "Dex5V9P4P5P1ContinuousClosureCycleAlignedApproachScale0035PhysicsEnvCfg",
    ),
    (
        "Dex5-V9P4-P5-P1-ContinuousClosureCycleAlignedServo002-Direct-v0",
        "Dex5V9P4P5P1ContinuousClosureCycleAlignedServo002PhysicsEnvCfg",
    ),
    (
        "Dex5-V9P4-P5-P1-ContinuousClosureCycleAlignedServo005-Direct-v0",
        "Dex5V9P4P5P1ContinuousClosureCycleAlignedServo005PhysicsEnvCfg",
    ),
):
    gym.register(
        id=task_id,
        entry_point=(
            f"{__name__}.p5_p1_physics_env:"
            "Dex5V9P4P5P1ContinuousTriadTrainPhysicsEnv"
        ),
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                f"{__name__}.p5_p1_physics_env:{cfg_name}"
            ),
            "skrl_cfg_entry_point": (
                f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml"
            ),
        },
    )

gym.register(
    id="Dex5-BrakedP4-Direct-v0",
    entry_point=f"{__name__}.braked_p4_physics_env:Dex5BrakedP4PhysicsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.braked_p4_physics_env:Dex5BrakedP4PhysicsEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml",
    },
)

gym.register(
    id="Dex5-P4Approach-Direct-v0",
    entry_point=f"{__name__}.braked_p4_physics_env:Dex5P4ApproachPhysicsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.braked_p4_physics_env:Dex5P4ApproachPhysicsEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml",
    },
)

gym.register(
    id="Dex5-CanonicalP3-P4Approach-Direct-v0",
    entry_point=(
        f"{__name__}.braked_p4_physics_env:Dex5CanonicalP3P4ApproachPhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.braked_p4_physics_env:"
            "Dex5CanonicalP3P4ApproachPhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml",
    },
)

gym.register(
    id="Dex5-CanonicalP3-P4Entry-Direct-v0",
    entry_point=f"{__name__}.braked_p4_physics_env:Dex5CanonicalP3P4EntryPhysicsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.braked_p4_physics_env:Dex5CanonicalP3P4EntryPhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml",
    },
)

gym.register(
    id="Dex5-CanonicalP3-Entry7ToP4Approach-Direct-v0",
    entry_point=(
        f"{__name__}.braked_p4_physics_env:Dex5CanonicalP3Entry7P4ApproachPhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.braked_p4_physics_env:"
            "Dex5CanonicalP3Entry7P4ApproachPhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml",
    },
)

gym.register(
    id="Dex5-CanonicalP3-Entry7ToP4Entry12-Direct-v0",
    entry_point=(
        f"{__name__}.braked_p4_physics_env:Dex5CanonicalP3Entry7P4Entry12PhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.braked_p4_physics_env:"
            "Dex5CanonicalP3Entry7P4Entry12PhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml",
    },
)

gym.register(
    id="Dex5-CanonicalEntry12-P4Approach20-Direct-v0",
    entry_point=(
        f"{__name__}.braked_p4_physics_env:Dex5CanonicalEntry12P4ApproachPhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.braked_p4_physics_env:"
            "Dex5CanonicalEntry12P4ApproachPhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml",
    },
)

gym.register(
    id="Dex5-CanonicalEntry20-P4Midway35-Direct-v0",
    entry_point=(
        f"{__name__}.braked_p4_physics_env:Dex5CanonicalEntry20P4MidwayPhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.braked_p4_physics_env:"
            "Dex5CanonicalEntry20P4MidwayPhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml",
    },
)

gym.register(
    id="Dex5-CanonicalEntry35-BrakedP4-Direct-v0",
    entry_point=(
        f"{__name__}.braked_p4_physics_env:Dex5CanonicalEntry35BrakedP4PhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.braked_p4_physics_env:Dex5CanonicalEntry35BrakedP4PhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml",
    },
)

gym.register(
    id="Dex5-CanonicalEntry35-P4Approach45-Direct-v0",
    entry_point=(
        f"{__name__}.braked_p4_physics_env:Dex5CanonicalEntry35P4Approach45PhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.braked_p4_physics_env:"
            "Dex5CanonicalEntry35P4Approach45PhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml",
    },
)

gym.register(
    id="Dex5-CanonicalEntry45-BrakedP4-Direct-v0",
    entry_point=(
        f"{__name__}.braked_p4_physics_env:Dex5CanonicalEntry45BrakedP4PhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.braked_p4_physics_env:Dex5CanonicalEntry45BrakedP4PhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml",
    },
)

gym.register(
    id="Dex5-CanonicalEntry45-BrakedP4At50-Direct-v0",
    entry_point=(
        f"{__name__}.braked_p4_physics_env:Dex5CanonicalEntry45BrakedP4At50PhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.braked_p4_physics_env:Dex5CanonicalEntry45BrakedP4At50PhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml",
    },
)

for phase_deg, hold_steps in ((47, 3), (48, 4), (49, 5), (50, 6)):
    gym.register(
        id=f"Dex5-CanonicalEntry45-PhaseCapture{phase_deg}-Direct-v0",
        entry_point=(
            f"{__name__}.braked_p4_physics_env:"
            f"Dex5CanonicalEntry45PhaseCapture{phase_deg}PhysicsEnv"
        ),
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                f"{__name__}.braked_p4_physics_env:"
                f"Dex5CanonicalEntry45PhaseCapture{phase_deg}PhysicsEnvCfg"
            ),
            "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml",
        },
    )

gym.register(
    id="Dex5-CanonicalEntry45-PhaseCapture50Noise-Direct-v0",
    entry_point=(
        f"{__name__}.braked_p4_physics_env:"
        "Dex5CanonicalEntry45PhaseCapture50NoisePhysicsEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.braked_p4_physics_env:"
            "Dex5CanonicalEntry45PhaseCapture50NoisePhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml",
    },
)

for curriculum_name in (
    "Late",
    "Mid",
    "Focus20",
    "Focus12",
    "Early12",
    "Early7",
    "All",
    "Focus0Gait",
    "Focus12Gait",
    "Early12Gait",
    "Early7Gait",
    "AllGait",
    "AllGaitFocus0X3",
):
    gym.register(
        id=f"Dex5-UnifiedP3-BrakedP4{curriculum_name}-Direct-v0",
        entry_point=(
            f"{__name__}.braked_p4_physics_env:"
            f"Dex5UnifiedP3BrakedP4{curriculum_name}PhysicsEnv"
        ),
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                f"{__name__}.braked_p4_physics_env:"
                f"Dex5UnifiedP3BrakedP4{curriculum_name}PhysicsEnvCfg"
            ),
            "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml",
        },
    )

gym.register(
    id="Dex5-CanonicalP3-P4Midway-Direct-v0",
    entry_point=f"{__name__}.braked_p4_physics_env:Dex5CanonicalP3P4MidwayPhysicsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.braked_p4_physics_env:Dex5CanonicalP3P4MidwayPhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml",
    },
)

gym.register(
    id="Dex5-CanonicalP3-BrakedP4-Direct-v0",
    entry_point=f"{__name__}.braked_p4_physics_env:Dex5CanonicalP3BrakedP4PhysicsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.braked_p4_physics_env:Dex5CanonicalP3BrakedP4PhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml",
    },
)

gym.register(
    id="Dex5-P4Midway-Direct-v0",
    entry_point=f"{__name__}.braked_p4_physics_env:Dex5P4MidwayPhysicsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.braked_p4_physics_env:Dex5P4MidwayPhysicsEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml",
    },
)

gym.register(
    id="Dex5-P4Bridge22-Direct-v0",
    entry_point=f"{__name__}.braked_p4_physics_env:Dex5P4Bridge22PhysicsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.braked_p4_physics_env:Dex5P4Bridge22PhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml",
    },
)

gym.register(
    id="Dex5-P4Bridge25-Direct-v0",
    entry_point=f"{__name__}.braked_p4_physics_env:Dex5P4Bridge25PhysicsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.braked_p4_physics_env:Dex5P4Bridge25PhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml",
    },
)

gym.register(
    id="Dex5-P4Bridge22Cache-Direct-v0",
    entry_point=f"{__name__}.braked_p4_physics_env:Dex5P4Bridge22CachePhysicsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.braked_p4_physics_env:Dex5P4Bridge22CachePhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml",
    },
)

gym.register(
    id="Dex5-P4Bridge22ThumbPreserve-Direct-v0",
    entry_point=f"{__name__}.braked_p4_physics_env:Dex5P4Bridge22ThumbPreservePhysicsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.braked_p4_physics_env:Dex5P4Bridge22ThumbPreservePhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml",
    },
)

gym.register(
    id="Dex5-P4Bridge22DelayedContact-Direct-v0",
    entry_point=f"{__name__}.braked_p4_physics_env:Dex5P4Bridge22DelayedContactPhysicsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.braked_p4_physics_env:Dex5P4Bridge22DelayedContactPhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml",
    },
)

gym.register(
    id="Dex5-P4Bridge22DelayedContactHold-Direct-v0",
    entry_point=f"{__name__}.braked_p4_physics_env:Dex5P4Bridge22DelayedContactHoldPhysicsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.braked_p4_physics_env:Dex5P4Bridge22DelayedContactHoldPhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml",
    },
)

gym.register(
    id="Dex5-P4Bridge22DelayedContactStatic-Direct-v0",
    entry_point=f"{__name__}.braked_p4_physics_env:Dex5P4Bridge22DelayedContactStaticPhysicsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.braked_p4_physics_env:Dex5P4Bridge22DelayedContactStaticPhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml",
    },
)

gym.register(
    id="Dex5-P4Bridge22DelayedContactProcessed-Direct-v0",
    entry_point=f"{__name__}.braked_p4_physics_env:Dex5P4Bridge22DelayedContactProcessedPhysicsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.braked_p4_physics_env:Dex5P4Bridge22DelayedContactProcessedPhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml",
    },
)

gym.register(
    id="Dex5-P4Bridge30-Direct-v0",
    entry_point=f"{__name__}.braked_p4_physics_env:Dex5P4Bridge30PhysicsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.braked_p4_physics_env:Dex5P4Bridge30PhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml",
    },
)

gym.register(
    id="Dex5-P4RecaptureHold-Direct-v0",
    entry_point=f"{__name__}.braked_p4_physics_env:Dex5P4RecaptureHoldPhysicsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.braked_p4_physics_env:Dex5P4RecaptureHoldPhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml",
    },
)

for task_id, cfg_name, env_name in (
    (
        "Dex5-V9P3-BrakedP4-Direct-v0",
        "Dex5V9P3BrakedP4PhysicsEnvCfg",
        "Dex5V9P3BrakedP4PhysicsEnv",
    ),
    (
        "Dex5-V9P3-BrakedP4Noise-Direct-v0",
        "Dex5V9P3BrakedP4NoisePhysicsEnvCfg",
        "Dex5V9P3BrakedP4NoisePhysicsEnv",
    ),
    (
        "Dex5-V9DualEntry0-45-Braked50-Direct-v0",
        "Dex5V9DualEntryBraked50PhysicsEnvCfg",
        "Dex5V9DualEntryBraked50PhysicsEnv",
    ),
    (
        "Dex5-V9DualEntry0-45-Long-Braked50-Direct-v0",
        "Dex5V9DualEntryLongBraked50PhysicsEnvCfg",
        "Dex5V9DualEntryLongBraked50PhysicsEnv",
    ),
    (
        "Dex5-V9AllStage-Braked50-Direct-v0",
        "Dex5V9AllStageBraked50PhysicsEnvCfg",
        "Dex5V9AllStageBraked50PhysicsEnv",
    ),
    *(
        (
            f"Dex5-V9BackwardSuffix{stage}-Braked50-Direct-v0",
            f"Dex5V9BackwardSuffix{stage}PhysicsEnvCfg",
            f"Dex5V9BackwardSuffix{stage}PhysicsEnv",
        )
        for stage in (40, 35, 30, 25, 20, 7, 0)
    ),
    (
        "Dex5-V9BackwardSuffix35Focus-Braked50-Direct-v0",
        "Dex5V9BackwardSuffix35FocusPhysicsEnvCfg",
        "Dex5V9BackwardSuffix35FocusPhysicsEnv",
    ),
    (
        "Dex5-V9P3-Entry7-Direct-v0",
        "Dex5V9P3Entry7PhysicsEnvCfg",
        "Dex5V9P3Entry7PhysicsEnv",
    ),
    (
        "Dex5-V9P3-Entry7Wide-Direct-v0",
        "Dex5V9P3Entry7WidePhysicsEnvCfg",
        "Dex5V9P3Entry7WidePhysicsEnv",
    ),
    (
        "Dex5-V9Entry7-Approach20-Direct-v0",
        "Dex5V9Entry7Approach20PhysicsEnvCfg",
        "Dex5V9Entry7Approach20PhysicsEnv",
    ),
    (
        "Dex5-V9Approach20-Mid35-Direct-v0",
        "Dex5V9Approach20Mid35PhysicsEnvCfg",
        "Dex5V9Approach20Mid35PhysicsEnv",
    ),
    (
        "Dex5-V9Approach20-Entry25-Direct-v0",
        "Dex5V9Approach20Entry25PhysicsEnvCfg",
        "Dex5V9Approach20Entry25PhysicsEnv",
    ),
    (
        "Dex5-V9Formal4096Approach20-Entry25-Direct-v0",
        "Dex5V9Formal4096Approach20Entry25PhysicsEnvCfg",
        "Dex5V9Approach20Entry25PhysicsEnv",
    ),
    (
        "Dex5-V9LongHistory7Approach20Hard5x-Entry25-Direct-v0",
        "Dex5V9LongHistory7Approach20Hard5xEntry25PhysicsEnvCfg",
        "Dex5V9LongHistory35Approach49PrefillBraked50PhysicsEnv",
    ),
    (
        "Dex5-V9Entry25-Mid30-Direct-v0",
        "Dex5V9Entry25Mid30PhysicsEnvCfg",
        "Dex5V9Entry25Mid30PhysicsEnv",
    ),
    (
        "Dex5-V9Mid30-Mid35-Direct-v0",
        "Dex5V9Mid30Mid35PhysicsEnvCfg",
        "Dex5V9Mid30Mid35PhysicsEnv",
    ),
    (
        "Dex5-V9Mid35-Approach40-Direct-v0",
        "Dex5V9Mid35Approach40PhysicsEnvCfg",
        "Dex5V9Mid35Approach40PhysicsEnv",
    ),
    (
        "Dex5-V9Approach40-Entry45-Direct-v0",
        "Dex5V9Approach40Entry45PhysicsEnvCfg",
        "Dex5V9Approach40Entry45PhysicsEnv",
    ),
    (
        "Dex5-V9ContinuousMid35-Approach40-Direct-v0",
        "Dex5V9ContinuousMid35Approach40PhysicsEnvCfg",
        "Dex5V9ContinuousMid35Approach40PhysicsEnv",
    ),
    (
        "Dex5-V9ContinuousApproach40-Entry45-Direct-v0",
        "Dex5V9ContinuousApproach40Entry45PhysicsEnvCfg",
        "Dex5V9ContinuousApproach40Entry45PhysicsEnv",
    ),
    (
        "Dex5-V9Approach40-Bridge42-Direct-v0",
        "Dex5V9Approach40Bridge42PhysicsEnvCfg",
        "Dex5V9Approach40Bridge42PhysicsEnv",
    ),
    (
        "Dex5-V9Cascade40-Bridge42-Direct-v0",
        "Dex5V9Cascade40Bridge42PhysicsEnvCfg",
        "Dex5V9Cascade40Bridge42PhysicsEnv",
    ),
    (
        "Dex5-V9Approach40-Bridge42Short-Direct-v0",
        "Dex5V9Approach40Bridge42ShortPhysicsEnvCfg",
        "Dex5V9Approach40Bridge42ShortPhysicsEnv",
    ),
    (
        "Dex5-V9Approach40-Bridge42Bootstrap-Direct-v0",
        "Dex5V9Approach40Bridge42BootstrapPhysicsEnvCfg",
        "Dex5V9Approach40Bridge42BootstrapPhysicsEnv",
    ),
    (
        "Dex5-V9Approach40-Bridge42Teacher-Direct-v0",
        "Dex5V9Approach40Bridge42TeacherPhysicsEnvCfg",
        "Dex5V9Approach40Bridge42TeacherPhysicsEnv",
    ),
    (
        "Dex5-V9Bridge42-StrictCache-Direct-v0",
        "Dex5V9Bridge42StrictCachePhysicsEnvCfg",
        "Dex5V9Bridge42StrictCachePhysicsEnv",
    ),
    (
        "Dex5-V9Bridge42-StrictHold6Teacher-Direct-v0",
        "Dex5V9Bridge42StrictHold6TeacherPhysicsEnvCfg",
        "Dex5V9Bridge42StrictHold6TeacherPhysicsEnv",
    ),
    (
        "Dex5-V9Bridge42-Entry45-Direct-v0",
        "Dex5V9Bridge42Entry45PhysicsEnvCfg",
        "Dex5V9Bridge42Entry45PhysicsEnv",
    ),
    (
        "Dex5-V9Cascade42-Entry45-Direct-v0",
        "Dex5V9Cascade42Entry45PhysicsEnvCfg",
        "Dex5V9Cascade42Entry45PhysicsEnv",
    ),
    (
        "Dex5-V9Cascade42-Entry45Teacher-Direct-v0",
        "Dex5V9Cascade42Entry45TeacherPhysicsEnvCfg",
        "Dex5V9Cascade42Entry45TeacherPhysicsEnv",
    ),
    (
        "Dex5-V9Cascade42-Entry43-Direct-v0",
        "Dex5V9Cascade42Entry43PhysicsEnvCfg",
        "Dex5V9Cascade42Entry43PhysicsEnv",
    ),
    (
        "Dex5-V9Cascade43-Entry45-Direct-v0",
        "Dex5V9Cascade43Entry45PhysicsEnvCfg",
        "Dex5V9Cascade43Entry45PhysicsEnv",
    ),
    (
        "Dex5-V9Bridge42-Entry45Teacher-Direct-v0",
        "Dex5V9Bridge42Entry45TeacherPhysicsEnvCfg",
        "Dex5V9Bridge42Entry45TeacherPhysicsEnv",
    ),
    (
        "Dex5-V9Entry45-Braked50-Direct-v0",
        "Dex5V9Entry45Braked50PhysicsEnvCfg",
        "Dex5V9Entry45Braked50PhysicsEnv",
    ),
    (
        "Dex5-V9CurrentUpstream45-Braked50-Direct-v0",
        "Dex5V9CurrentUpstream45Braked50PhysicsEnvCfg",
        "Dex5V9CurrentUpstream45Braked50PhysicsEnv",
    ),
    (
        "Dex5-V9CurrentUpstream45-Approach49-Direct-v0",
        "Dex5V9CurrentUpstream45Approach49PhysicsEnvCfg",
        "Dex5V9CurrentUpstream45Approach49PhysicsEnv",
    ),
    (
        "Dex5-V9CurrentApproach49Entry-Braked50-Direct-v0",
        "Dex5V9CurrentApproach49EntryBraked50PhysicsEnvCfg",
        "Dex5V9CurrentApproach49EntryBraked50PhysicsEnv",
    ),
    (
        "Dex5-V9Approach49Prefill-Braked50-Direct-v0",
        "Dex5V9Approach49PrefillBraked50PhysicsEnvCfg",
        "Dex5V9Approach49PrefillBraked50PhysicsEnv",
    ),
    (
        "Dex5-V9Formal4096Approach49Prefill-Braked50-Direct-v0",
        "Dex5V9Formal4096Approach49PrefillBraked50PhysicsEnvCfg",
        "Dex5V9Approach49PrefillBraked50PhysicsEnv",
    ),
    (
        "Dex5-V9Formal4096Approach49PrefillLowpen325-Braked50-Direct-v0",
        "Dex5V9Formal4096Approach49PrefillLowpen325Braked50PhysicsEnvCfg",
        "Dex5V9Approach49PrefillBraked50PhysicsEnv",
    ),
    (
        "Dex5-V9Approach49PrefillCore5xFormal541Mix-Braked50-Direct-v0",
        "Dex5V9Approach49PrefillCore5xFormal541MixBraked50PhysicsEnvCfg",
        "Dex5V9Approach49PrefillBraked50PhysicsEnv",
    ),
    (
        "Dex5-V9LongHistory35Approach49Prefill-Braked50-Direct-v0",
        "Dex5V9LongHistory35Approach49PrefillBraked50PhysicsEnvCfg",
        "Dex5V9LongHistory35Approach49PrefillBraked50PhysicsEnv",
    ),
    (
        "Dex5-V9LongHistory35Approach49Hard3xPrefill-Braked50-Direct-v0",
        "Dex5V9LongHistory35Approach49Hard3xPrefillBraked50PhysicsEnvCfg",
        "Dex5V9LongHistory35Approach49PrefillBraked50PhysicsEnv",
    ),
    (
        "Dex5-V9Current8345LongHistory35Approach49Hard5x-Braked50-Direct-v0",
        "Dex5V9Current8345LongHistory35Approach49Hard5xBraked50PhysicsEnvCfg",
        "Dex5V9LongHistory35Approach49PrefillBraked50PhysicsEnv",
    ),
    (
        "Dex5-V9Entry45-Braked50Teacher-Direct-v0",
        "Dex5V9Entry45Braked50TeacherPhysicsEnvCfg",
        "Dex5V9Entry45Braked50TeacherPhysicsEnv",
    ),
    (
        "Dex5-V9Entry45-Approach47-Direct-v0",
        "Dex5V9Entry45Approach47PhysicsEnvCfg",
        "Dex5V9Entry45Approach47PhysicsEnv",
    ),
    (
        "Dex5-V9Entry45-Approach47Teacher-Direct-v0",
        "Dex5V9Entry45Approach47TeacherPhysicsEnvCfg",
        "Dex5V9Entry45Approach47TeacherPhysicsEnv",
    ),
    (
        "Dex5-V9Entry45-Approach47Hard-Direct-v0",
        "Dex5V9Entry45Approach47HardPhysicsEnvCfg",
        "Dex5V9Entry45Approach47HardPhysicsEnv",
    ),
    (
        "Dex5-V9Entry45-HardCapture44-Direct-v0",
        "Dex5V9Entry45HardCapture44PhysicsEnvCfg",
        "Dex5V9Entry45HardCapture44PhysicsEnv",
    ),
    (
        "Dex5-V9Entry45-ThumbSupportCorrection-Direct-v0",
        "Dex5V9Entry45ThumbSupportCorrectionPhysicsEnvCfg",
        "Dex5V9Entry45ThumbSupportCorrectionPhysicsEnv",
    ),
    (
        "Dex5-V9Cascade-ThumbSupportCorrection-Direct-v0",
        "Dex5V9CascadeThumbSupportCorrectionPhysicsEnvCfg",
        "Dex5V9CascadeThumbSupportCorrectionPhysicsEnv",
    ),
    (
        "Dex5-V9ThumbSupportedEntry-Approach47-Direct-v0",
        "Dex5V9ThumbSupportedEntryApproach47PhysicsEnvCfg",
        "Dex5V9ThumbSupportedEntryApproach47PhysicsEnv",
    ),
    (
        "Dex5-V9ThumbSupportedEntry-Capture45-Direct-v0",
        "Dex5V9ThumbSupportedEntryCapture45PhysicsEnvCfg",
        "Dex5V9ThumbSupportedEntryCapture45PhysicsEnv",
    ),
    (
        "Dex5-V9CascadeSupported-Capture45-Direct-v0",
        "Dex5V9CascadeSupportedTo45PhysicsEnvCfg",
        "Dex5V9CascadeSupportedTo45PhysicsEnv",
    ),
    (
        "Dex5-V9CascadeSupported-Approach43-Direct-v0",
        "Dex5V9CascadeSupportedApproach43PhysicsEnvCfg",
        "Dex5V9CascadeSupportedApproach43PhysicsEnv",
    ),
    (
        "Dex5-V9CascadeSupported-Approach435-Direct-v0",
        "Dex5V9CascadeSupportedApproach435PhysicsEnvCfg",
        "Dex5V9CascadeSupportedApproach435PhysicsEnv",
    ),
    (
        "Dex5-V9Cascade45-Approach47-Direct-v0",
        "Dex5V9Cascade45Approach47PhysicsEnvCfg",
        "Dex5V9Cascade45Approach47PhysicsEnv",
    ),
    (
        "Dex5-V9ThumbSupportedCapture45-Approach47-Direct-v0",
        "Dex5V9ThumbSupportedCapture45Approach47PhysicsEnvCfg",
        "Dex5V9ThumbSupportedCapture45Approach47PhysicsEnv",
    ),
    (
        "Dex5-V9ThumbSupportedApproach47-Capture48-Direct-v0",
        "Dex5V9ThumbSupportedApproach47Capture48PhysicsEnvCfg",
        "Dex5V9ThumbSupportedApproach47Capture48PhysicsEnv",
    ),
    (
        "Dex5-V9Cascade47-Capture48-Direct-v0",
        "Dex5V9Cascade47Capture48PhysicsEnvCfg",
        "Dex5V9Cascade47Capture48PhysicsEnv",
    ),
    (
        "Dex5-V9ThumbSupportedCapture48-Braked50-Direct-v0",
        "Dex5V9ThumbSupportedCapture48Braked50PhysicsEnvCfg",
        "Dex5V9ThumbSupportedCapture48Braked50PhysicsEnv",
    ),
    (
        "Dex5-V9Cascade-BrakedP4-Direct-v0",
        "Dex5V9CascadeBrakedP4PhysicsEnvCfg",
        "Dex5V9CascadeBrakedP4PhysicsEnv",
    ),
    (
        "Dex5-V9Mid35-Braked50-Direct-v0",
        "Dex5V9Mid35Braked50PhysicsEnvCfg",
        "Dex5V9Mid35Braked50PhysicsEnv",
    ),
):
    gym.register(
        id=task_id,
        entry_point=f"{__name__}.braked_p4_physics_env:{env_name}",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.braked_p4_physics_env:{cfg_name}",
            "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_recapture_p3_cfg.yaml",
        },
    )

gym.register(
    id="Dex5-IndexNearSurface-Direct-v0",
    entry_point=f"{__name__}.index_near_surface_physics_env:Dex5IndexNearSurfacePhysicsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.index_near_surface_physics_env:Dex5IndexNearSurfacePhysicsEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_physics_cfg.yaml",
    },
)
