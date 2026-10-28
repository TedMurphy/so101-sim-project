"""apply_robot_recolor.py — Recolour the SO-101 robot arm to off-white/grey.

Drop this file alongside apply_graphpaper_texture.py in the scripts folder.
Import and call apply_robot_recolor(stage) after env.reset() and before
timeline.play().

Strategy (v3 — direct shader patch):
    stage.TraverseAll() finds no Mesh prims under the robot because Isaac Lab
    stores geometry in USD references/payloads whose Mesh children never appear
    in the flat traversal.  However the Looks scope IS visible, and it contains
    the real OmniPBR/MDL shaders that drive the robot's colour.  We patch their
    inputs directly — no new material, no binding required.

    The yellow body is driven by the 'material_a_3d_printed' shader.  We target
    that prim (and 'DefaultMaterial' as a secondary) and overwrite:
        inputs:diffuse_color_constant  →  off-white/grey  (OmniPBR)
        inputs:emissive_color          →  black           (OmniPBR)
        inputs:enable_emission         →  False
        inputs:emissive_intensity      →  0.0
    UsdPreviewSurface names (diffuseColor / emissiveColor) are tried as a fallback
    so the function works even if the USD is later updated to a different shader.
"""

# Materials to repaint (by prim name under Robot/Looks/).
# All others (servos, PCB, lens, anodised hardware) are left as-is.
_REPAINT = {"material_a_3d_printed", "DefaultMaterial"}

# Off-white with a faint grey-blue tint — matches the real SO-101 aluminium arm.
# Values are linear sRGB.
_DIFFUSE   = (0.65, 0.65, 0.68)
_METALLIC  = 0.15
_ROUGHNESS = 0.55


def apply_robot_recolor(stage, robot_root="/World/envs/env_0/Robot"):
    """
    Patch existing OmniPBR/MDL shaders in Robot/Looks/ so the arm body is
    rendered off-white instead of yellow.

    Parameters
    ----------
    stage      : pxr.Usd.Stage  — from omni.usd.get_context().get_stage()
    robot_root : str            — prim path of the robot root in the stage

    Returns
    -------
    int  — number of shaders successfully patched
    """
    from pxr import UsdShade, Sdf, Gf

    # ── Verify robot root ────────────────────────────────────────────────────
    root_prim = stage.GetPrimAtPath(robot_root)
    if not root_prim.IsValid():
        for alt in ["/World/envs/env_0/robot",
                    "/World/envs/env_0/so101",
                    "/World/envs/env_0/SO101"]:
            p = stage.GetPrimAtPath(alt)
            if p.IsValid():
                robot_root = alt
                root_prim  = p
                print(f"[RECOLOR] Using alternate robot root: {robot_root}")
                break
        if not root_prim.IsValid():
            print(f"[RECOLOR] ERROR: robot root not found at {robot_root}")
            return 0

    looks_path = robot_root + "/Looks"
    looks_prim = stage.GetPrimAtPath(looks_path)
    if not looks_prim.IsValid():
        print(f"[RECOLOR] ERROR: {looks_path} not found in stage")
        return 0

    diffuse_gf = Gf.Vec3f(*_DIFFUSE)
    black_gf   = Gf.Vec3f(0.0, 0.0, 0.0)

    patched = 0

    for mat_prim in looks_prim.GetChildren():
        mat_name = mat_prim.GetName()
        if mat_name not in _REPAINT:
            continue

        # Walk direct children for the Shader prim (may be named anything)
        for child in mat_prim.GetChildren():
            if child.GetTypeName() != "Shader":
                continue

            shader_path = str(child.GetPath())
            changed = False

            # ── OmniPBR / MDL attribute names ────────────────────────────────
            def _set_attr(name, val):
                attr = child.GetAttribute(name)
                if attr and attr.IsValid():
                    attr.Set(val)
                    return True
                # Attribute may not exist yet — create it
                try:
                    from pxr import Sdf as _Sdf
                    if isinstance(val, Gf.Vec3f):
                        attr = child.CreateAttribute(name, _Sdf.ValueTypeNames.Color3f)
                    elif isinstance(val, float):
                        attr = child.CreateAttribute(name, _Sdf.ValueTypeNames.Float)
                    elif isinstance(val, bool):
                        attr = child.CreateAttribute(name, _Sdf.ValueTypeNames.Bool)
                    else:
                        return False
                    attr.Set(val)
                    return True
                except Exception:
                    return False

            # Diffuse / albedo
            if _set_attr("inputs:diffuse_color_constant", diffuse_gf):
                changed = True
            if _set_attr("inputs:albedo_add", Gf.Vec3f(0.0, 0.0, 0.0)):
                pass  # tint override — zero means no additive tint

            # Kill emissive (yellow glow is often caused by a non-zero emissive)
            _set_attr("inputs:emissive_color",     black_gf)
            _set_attr("inputs:emissive_intensity",  0.0)
            _set_attr("inputs:enable_emission",     False)

            # Metallic / roughness
            _set_attr("inputs:metallic_constant",               _METALLIC)
            _set_attr("inputs:reflection_roughness_constant",   _ROUGHNESS)

            # ── UsdPreviewSurface names (fallback) ────────────────────────────
            if not changed:
                shader = UsdShade.Shader(child)
                inp = shader.GetInput("diffuseColor")
                if not inp:
                    inp = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
                inp.Set(diffuse_gf)

                em = shader.GetInput("emissiveColor")
                if not em:
                    em = shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f)
                em.Set(black_gf)

                mt = shader.GetInput("metallic")
                if not mt:
                    mt = shader.CreateInput("metallic", Sdf.ValueTypeNames.Float)
                mt.Set(_METALLIC)

                rg = shader.GetInput("roughness")
                if not rg:
                    rg = shader.CreateInput("roughness", Sdf.ValueTypeNames.Float)
                rg.Set(_ROUGHNESS)

                changed = True
                print(f"[RECOLOR] UsdPreviewSurface fallback used for {shader_path}")

            if changed:
                patched += 1
                print(f"[RECOLOR] ✓ Patched shader: {shader_path}")
            else:
                print(f"[RECOLOR] WARNING: could not patch {shader_path}")
                for attr in child.GetAttributes():
                    print(f"[RECOLOR]   {attr.GetName()} = {attr.Get()}")

    if patched:
        print(f"[RECOLOR] ✓ {patched} shader(s) patched to off-white/grey")
    else:
        print(f"[RECOLOR] WARNING: 0 shaders patched — arm stays default colour")
        # Diagnostic: list everything under Looks
        for p in stage.TraverseAll():
            ps = str(p.GetPath())
            if ps.startswith(looks_path):
                print(f"[RECOLOR]   {ps}  type={p.GetTypeName()!r}")

    return patched