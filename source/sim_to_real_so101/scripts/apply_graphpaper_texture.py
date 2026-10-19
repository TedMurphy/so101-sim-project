#!/usr/bin/env python3
"""
apply_graphpaper_texture.py
---------------------------
Apply a graph-paper texture to the SIM mat so it matches the real overhead camera view.

Works with both UsdPreviewSurface (standard USD) and OmniPBR (MDL) shaders.

Usage — call from within any running Isaac Sim script (after env.reset()):

    from apply_graphpaper_texture import apply_graphpaper_texture
    import omni.usd
    apply_graphpaper_texture(
        omni.usd.get_context().get_stage(),
        texture_path="/home/ted/so101-sim-project/graphpaper_mat.png",
    )
"""

from __future__ import annotations

# ── Configure: where is the mat prim? ─────────────────────────────────────────
_MAT_PRIM_CANDIDATES = [
    "/World/envs/env_0/mat",
    "/World/envs/env_0/Mat",
    "/World/mat",
    "/World/Mat",
]

# ── UV layout (must match vertex ORDER in mat.usda points array) ───────────────
# mat.usda point order (local coords):
#   0: ( 0.375,  0.210, 0.003)  near-left
#   1: (-0.375,  0.210, 0.003)  near-right
#   2: (-0.350, -0.170, 0.003)  far-right
#   3: ( 0.350, -0.195, 0.003)  far-left
#
# Camera is Rz(180°) overhead: image-left = Isaac +X (near-left vertex)
# Map corners → texture corners:
#   near-left  → UV (0, 0)   top-left of texture
#   near-right → UV (1, 0)   top-right
#   far-right  → UV (1, 1)   bottom-right
#   far-left   → UV (0, 1)   bottom-left
_UV_COORDS = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]

# ─────────────────────────────────────────────────────────────────────────────


def _wire_preview_surface_texture(stage, shader_path: str, texture_path: str) -> bool:
    """
    Wire a UsdUVTexture + UsdPrimvarReader_float2 into an existing UsdPreviewSurface.

    Standard USD shading network:
        UVReader(st primvar) → TexReader(file=png).rgb → Shader.diffuseColor
    """
    from pxr import UsdShade, Sdf

    mat_path      = str(Sdf.Path(shader_path).GetParentPath())
    uv_reader_path = f"{mat_path}/UVReader"
    tex_reader_path = f"{mat_path}/TexReader"

    # ── UV primvar reader ──────────────────────────────────────────────────────
    uv_reader = UsdShade.Shader.Define(stage, uv_reader_path)
    uv_reader.CreateIdAttr("UsdPrimvarReader_float2")
    uv_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    uv_out = uv_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    print(f"[TEXTURE] Created UVReader at {uv_reader_path}")

    # ── Texture file reader ────────────────────────────────────────────────────
    tex_reader = UsdShade.Shader.Define(stage, tex_reader_path)
    tex_reader.CreateIdAttr("UsdUVTexture")
    tex_reader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture_path))
    tex_reader.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
    tex_reader.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
    # Feed UV coords into the texture reader
    tex_reader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(uv_out)
    rgb_out = tex_reader.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    print(f"[TEXTURE] Created TexReader at {tex_reader_path}")

    # ── Connect rgb → diffuseColor on the surface shader ─────────────────────
    surf_shader = UsdShade.Shader(stage.GetPrimAtPath(shader_path))
    surf_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(rgb_out)
    surf_shader.CreateInput("roughness",    Sdf.ValueTypeNames.Float).Set(0.85)
    surf_shader.CreateInput("metallic",     Sdf.ValueTypeNames.Float).Set(0.0)
    print(f"[TEXTURE] Connected TexReader.rgb → UsdPreviewSurface.diffuseColor")
    return True


def _wire_omnipbr_texture(stage, shader_path: str, texture_path: str) -> bool:
    """Set diffuse_texture on an existing OmniPBR MDL shader via kit commands."""
    from pxr import UsdShade, Sdf
    try:
        import omni.kit.commands
        omni.kit.commands.execute(
            "ChangeProperty",
            prop_path=f"{shader_path}.inputs:diffuse_texture",
            value=Sdf.AssetPath(texture_path),
            prev=Sdf.AssetPath(""),
        )
        UsdShade.Shader(stage.GetPrimAtPath(shader_path)).CreateInput(
            "project_uvw", Sdf.ValueTypeNames.Bool).Set(False)
        print(f"[TEXTURE] Set OmniPBR diffuse_texture via ChangeProperty")
        return True
    except Exception as e:
        print(f"[TEXTURE] OmniPBR ChangeProperty failed: {e}")
        return False


def apply_graphpaper_texture(stage, texture_path: str, mat_prim_path: str | None = None) -> bool:
    """
    Apply graph-paper texture to the mat in the current Isaac Sim stage.

    Args:
        stage:          The USD stage (from omni.usd.get_context().get_stage()).
        texture_path:   Absolute path to graphpaper_mat.png on disk.
        mat_prim_path:  Override the mat prim path (default: auto-detect).

    Returns True on success, False on failure.
    """
    import os

    try:
        from pxr import UsdShade, UsdGeom, Sdf, Vt
    except ImportError:
        print("[TEXTURE] pxr not available — is this running inside Isaac Sim?")
        return False

    # ── 0. Verify texture file ─────────────────────────────────────────────────
    texture_path = os.path.abspath(texture_path)
    if not os.path.isfile(texture_path):
        print(f"[TEXTURE] ERROR: texture file not found: {texture_path}")
        return False
    print(f"[TEXTURE] Texture file OK: {texture_path} ({os.path.getsize(texture_path):,} bytes)")

    # ── 1. Find the mat prim ───────────────────────────────────────────────────
    mat_prim = None
    candidates = [mat_prim_path] if mat_prim_path else _MAT_PRIM_CANDIDATES
    for path in candidates:
        if path is None:
            continue
        p = stage.GetPrimAtPath(path)
        if p.IsValid():
            mat_prim = p
            print(f"[TEXTURE] Found mat prim at {path}")
            break

    if mat_prim is None:
        for prim in stage.TraverseAll():
            if prim.GetName().lower() == "mat" and prim.GetTypeName() == "Mesh":
                mat_prim = prim
                print(f"[TEXTURE] Found mat prim via traversal at {prim.GetPath()}")
                break

    if mat_prim is None:
        print("[TEXTURE] ERROR: Could not find mat prim.")
        return False

    # ── 2. Find existing material / shader ────────────────────────────────────
    existing_binding    = UsdShade.MaterialBindingAPI(mat_prim).GetDirectBinding()
    existing_mat_path   = None
    existing_shdr_path  = None
    existing_shader_id  = None

    if existing_binding and existing_binding.GetMaterial():
        existing_mat_path = str(existing_binding.GetMaterialPath())
        print(f"[TEXTURE] Existing material: {existing_mat_path}")
        existing_mat_prim = stage.GetPrimAtPath(existing_mat_path)
        if existing_mat_prim.IsValid():
            for child in existing_mat_prim.GetChildren():
                shdr = UsdShade.Shader(child)
                if shdr:
                    existing_shdr_path = str(child.GetPath())
                    existing_shader_id = shdr.GetShaderId()
                    src = shdr.GetSourceAsset("mdl")
                    print(f"[TEXTURE] Existing shader: {existing_shdr_path}  "
                          f"id={existing_shader_id}  src={src}")
                    break
    else:
        print("[TEXTURE] No existing material binding.")

    # ── 3. Wire texture into the shader ───────────────────────────────────────
    if existing_shdr_path and existing_shader_id == "UsdPreviewSurface":
        # ── Standard USD path ─────────────────────────────────────────────────
        tex_ok = _wire_preview_surface_texture(stage, existing_shdr_path, texture_path)

    elif existing_shdr_path and "OmniPBR" in (existing_shader_id or ""):
        # ── OmniPBR MDL path ──────────────────────────────────────────────────
        tex_ok = _wire_omnipbr_texture(stage, existing_shdr_path, texture_path)

    else:
        # ── No usable existing shader: create a fresh OmniPBR ─────────────────
        looks_scope  = "/World/Looks"
        mat_usd_path = f"{looks_scope}/GraphPaperMat"
        shader_path  = f"{mat_usd_path}/Shader"

        if not stage.GetPrimAtPath(looks_scope).IsValid():
            stage.DefinePrim(looks_scope, "Scope")

        material = UsdShade.Material.Define(stage, mat_usd_path)
        shader   = UsdShade.Shader.Define(stage, shader_path)
        shader.SetSourceAsset("OmniPBR.mdl", "mdl")
        shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
        shader.CreateIdAttr("OmniPBR")
        shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(0.85)
        shader.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(0.0)

        mdl_out = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput("mdl").ConnectToSource(mdl_out)
        try:
            material.CreateVolumeOutput("mdl").ConnectToSource(mdl_out)
            material.CreateDisplacementOutput("mdl").ConnectToSource(mdl_out)
        except Exception:
            pass

        tex_ok = _wire_omnipbr_texture(stage, shader_path, texture_path)
        if not tex_ok:
            shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(
                Sdf.AssetPath(texture_path))

        UsdShade.MaterialBindingAPI(mat_prim).Bind(
            material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        )
        existing_shdr_path = shader_path
        print(f"[TEXTURE] Created + bound new OmniPBR material at {mat_usd_path}")

    # ── 4. Set UV (st) primvar on the mesh ────────────────────────────────────
    uv_ok = False
    try:
        primvars_api = UsdGeom.PrimvarsAPI(mat_prim)
        st_primvar = primvars_api.CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        st_primvar.Set(Vt.Vec2fArray(_UV_COORDS))
        uv_ok = True
        print(f"[TEXTURE] Set UV primvar 'st' on {mat_prim.GetPath()}")
    except Exception as e:
        print(f"[TEXTURE] PrimvarsAPI failed: {e}")

    if not uv_ok:
        print("[TEXTURE] Warning: could not set explicit UVs.")

    print(f"[TEXTURE] ✓ Graph-paper texture applied to {mat_prim.GetPath()}")
    return True


# ── Standalone test (run inside Isaac Sim via ./isaaclab.sh -p) ───────────────
if __name__ == "__main__":
    import os, argparse
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    parser.add_argument("--texture", default=os.path.expanduser(
        "~/so101-sim-project/graphpaper_mat.png"))
    parser.add_argument("--task", default="Lerobot-So101-Teleop-Cylinder-To-Target")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(["--livestream", "2", "--enable_cameras"])
    args.device = "cpu"

    launcher = AppLauncher(args)
    sim_app   = launcher.app

    import omni.usd
    import gymnasium as gym
    import isaaclab_tasks                           # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    import sim_to_real_so101.tasks                  # noqa: F401

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=None, use_fabric=True)
    env     = gym.make(args.task, cfg=env_cfg)
    env.reset()
    for _ in range(10):
        sim_app.update()

    stage = omni.usd.get_context().get_stage()
    ok = apply_graphpaper_texture(stage, args.texture)
    if ok:
        print("[TEST] Texture applied — open http://localhost:8211/streaming/webrtc-demo/ to verify.")
    else:
        print("[TEST] Texture application failed — see errors above.")

    while sim_app.is_running():
        sim_app.update()

    env.close()
    sim_app.close()