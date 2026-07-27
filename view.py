#!/usr/bin/env python
"""
view.py — watch a water-droplet movie made by simulate.py.

    python view.py                    # opens water_droplet.npz
    python view.py my_run.npz

Controls:
    space   play / pause
    a       auto-camera on/off (it flies in close whenever a reaction
            is about to happen — off = normal mouse orbit and zoom)
    r       restart the movie
    mouse   drag to orbit, scroll to zoom (when auto-camera is off)

Colors never change: oxygen is always red, hydrogen is always white —
that is just what the atom IS.  Two extra things get drawn on top:

    yellow ring = these atoms are reacting RIGHT NOW
    orange  +   = H3O+, a molecule that picked up an extra hydrogen
    blue    -   = OH-, a molecule that lost a hydrogen
"""

import argparse

import numpy as np
import pyvista as pv

ATOM_COLOR = {"O": "#d33c32", "H": "#ebebeb"}   # an atom's colour NEVER changes
REACTING = (255, 230, 40, 230)  # (red, green, blue, how solid) — the yellow ring
CHARGE_LABEL = {         # the charge sign floating over a charged molecule.
    # "count" is how many hydrogens that oxygen is holding.  A dash is a much
    # thinner shape than a plus, so the minus gets a bigger font to match.
    "+": {"count": 3, "color": "#ff9600", "size": 34},   # H3O+ : holding 3
    "-": {"count": 1, "color": "#5aa0ff", "size": 60},   # OH-  : holding 1
}
TITLES = {
    "ionization": "IONIZATION!   H2O + H2O  ->  H3O+  +  OH-",
    "recombination": "RECOMBINATION!   H3O+ + OH-  ->  2 H2O",
    "hop": "PROTON HOP — the charge jumps to a neighbour",
}
LEAD, TAIL, RAMP = 30, 30, 18   # frames: fly in LEAD before a reaction,
                                # stay TAIL after, fade over RAMP.
FLASH = 14                      # a reaction glows yellow for this many frames...
FLASH_STARRING = 76             # ...unless the camera flew in to watch it, in
                                # which case it stays lit the whole close-up.


def load(path):
    data = np.load(path, allow_pickle=False)
    events = list(zip(data["event_frame"].tolist(),
                      data["event_kind"].tolist(),
                      data["event_atoms"].tolist()))
    return data["positions"], data["bonds"], data["elements"], float(data["radius"]), events


def precompute_glow(positions, events, shots):
    """Work out which atoms wear the yellow ring on every frame, up front.

    The last number is how solid the ring is: 0 means invisible, which is
    what an atom that is not reacting gets.
    """
    n_frames, n_atoms = positions.shape[0], positions.shape[1]
    starring = {frame for frame, _, _ in shots}
    react = np.zeros((n_frames, n_atoms, 4), dtype=np.uint8)
    for frame, _, atoms in events:  # light up the atoms doing the reaction
        hold = FLASH_STARRING if frame in starring else FLASH
        lo, hi = max(0, frame - hold // 2), min(n_frames, frame + hold)
        react[lo:hi, atoms] = REACTING
    return react


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return 3 * x ** 2 - 2 * x ** 3


def plan_shots(events):
    """Pick which reactions the camera will fly in on.

    Reactions can happen so often that their camera windows would all
    overlap and the camera would never pull back out.  So we film one
    reaction at a time, with a little breather in between — every other
    reaction still flashes yellow in the background.
    """
    shots, breather = [], 40
    for event in events:
        if not shots or event[0] - LEAD > shots[-1][0] + TAIL + breather:
            shots.append(event)
    return shots


def active_event(shots, frame):
    """The reaction (if any) the camera should be watching right now.

    Returns (event, weight): weight rises 0 -> 1 as the camera flies in,
    holds at 1 through the reaction, and falls back to 0 afterwards.
    """
    for ev_frame, kind, atoms in shots:
        if ev_frame - LEAD <= frame <= ev_frame + TAIL:
            w = min(smoothstep((frame - (ev_frame - LEAD)) / RAMP),
                    smoothstep(((ev_frame + TAIL) - frame) / RAMP))
            return (ev_frame, kind, atoms), w
    return None, 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("trajectory", nargs="?", default="water_droplet.npz")
    parser.add_argument("--snapshot", type=int, nargs="*", default=None,
                        help="save these frames as PNGs instead of opening a window")
    args = parser.parse_args()

    positions, bonds, elements, radius, events = load(args.trajectory)
    shots = plan_shots(events)
    react_glow = precompute_glow(positions, events, shots)
    n_frames = positions.shape[0]
    center = positions[0].mean(axis=0)
    print(f"🎬 {n_frames} frames, {len(events)} reactions "
          f"({len(shots)} camera close-ups). "
          f"space=pause  a=auto-camera  r=restart")

    o_mask, h_mask = elements == "O", elements == "H"
    pl = pv.Plotter(window_size=(1100, 800), off_screen=args.snapshot is not None)
    pl.set_background("#0a0a14")

    # Every atom is a real little ball (so it still looks right up close).
    # Trick: build ONE template sphere, stamp a copy at every atom, and each
    # frame just slide all the copies to the atoms' new positions.
    def ball_cloud(n_balls, radius):
        template = pv.Sphere(radius=radius, theta_resolution=14, phi_resolution=14)
        pts = np.tile(template.points, (n_balls, 1))
        faces = template.faces.reshape(-1, 4).copy()
        all_faces = np.vstack([
            faces + [0, k * template.n_points, k * template.n_points,
                     k * template.n_points] for k in range(n_balls)])
        return pv.PolyData(pts, all_faces), template.points, template.n_points

    def move_balls(mesh, template_pts, centers):
        mesh.points = (centers[:, None, :] + template_pts[None, :, :]).reshape(-1, 3)

    o_balls, o_template, o_npts = ball_cloud(int(o_mask.sum()), radius=0.45)
    h_balls, h_template, h_npts = ball_cloud(int(h_mask.sum()), radius=0.28)
    move_balls(o_balls, o_template, positions[0][o_mask].astype(float))
    move_balls(h_balls, h_template, positions[0][h_mask].astype(float))

    # The halos: bigger balls around the atoms.  We hide each halo's front
    # half (culling below), so all you see is a coloured RING poking out
    # around the atom — the atom's own colour is never painted over.
    # A plain water molecule's halo is fully invisible, so you only ever
    # see a glow where something interesting is happening.
    #
    def glow_layer(mask, radius, colours):
        mesh, template, npts = ball_cloud(int(mask.sum()), radius)
        move_balls(mesh, template, positions[0][mask].astype(float))
        mesh["rgba"] = np.repeat(colours[0][mask], npts, axis=0)
        pl.add_mesh(mesh, scalars="rgba", rgb=True, culling="front", ambient=0.6)
        return {"mesh": mesh, "template": template, "npts": npts,
                "mask": mask, "colours": colours}

    glow_layers = [
        glow_layer(o_mask, 0.85, react_glow),    # ring round a reacting oxygen
        glow_layer(h_mask, 0.50, react_glow),    # ...and the hopping hydrogen
    ]

    # A floating charge sign over every ion: "+" or "-".  We keep one sign
    # slot per oxygen forever and simply blank out the ones that are plain
    # water, so signs never have to be created or destroyed while playing.
    n_oxygens = int(o_mask.sum())
    label_lift = np.array([0.0, 0.0, 1.4])   # float the tag above the molecule
    label_layers = []
    for name, style in CHARGE_LABEL.items():
        cloud = pv.PolyData(positions[0][o_mask].astype(float))
        cloud["tag"] = [""] * n_oxygens
        pl.add_point_labels(cloud, "tag", font_size=style["size"], bold=True,
                            text_color=style["color"], show_points=False,
                            always_visible=True, shape=None)
        label_layers.append((cloud, name, style["count"]))

    def paint_labels(frame, o_at):
        counts = np.bincount(bonds[frame, :, 0] // 3, minlength=n_oxygens)
        for cloud, name, wanted in label_layers:
            cloud.points = o_at + label_lift
            cloud["tag"] = np.where(counts == wanted, name, "")

    def paint_glow(frame):
        for layer in glow_layers:
            mask, npts = layer["mask"], layer["npts"]
            move_balls(layer["mesh"], layer["template"],
                       positions[frame][mask].astype(float))
            layer["mesh"]["rgba"] = np.repeat(
                layer["colours"][frame][mask], npts, axis=0)

    wire = pv.PolyData(positions[0].astype(float))
    wire.lines = np.column_stack(
        [np.full(len(bonds[0]), 2), bonds[0]]).ravel()

    pl.add_mesh(wire, color="#999999", line_width=4, render_lines_as_tubes=True)
    pl.add_mesh(o_balls, color=ATOM_COLOR["O"], smooth_shading=True)
    pl.add_mesh(h_balls, color=ATOM_COLOR["H"], smooth_shading=True)
    hud = pl.add_text("", position="upper_left", font_size=11, color="white")

    state = {"frame": 0, "playing": True, "auto_cam": True, "bonds": bonds[0]}

    def place_camera(frame):
        event, w = active_event(shots, frame)
        target = center
        if event:
            _, _, atoms = event
            target = positions[frame, atoms].mean(axis=0)  # the reacting trio
        focal = center * (1 - w) + target * w
        distance = (4.0 * radius) * (1 - w) + 18.0 * w
        azim, elev = np.radians(frame * 0.35), np.radians(18)
        offset = np.array([np.cos(elev) * np.cos(azim),
                           np.cos(elev) * np.sin(azim), np.sin(elev)])
        pl.camera.position = focal + distance * offset
        pl.camera.focal_point = focal
        pl.camera.up = (0.0, 0.0, 1.0)

    def show_frame(frame):
        o_at = positions[frame][o_mask].astype(float)
        h_at = positions[frame][h_mask].astype(float)
        move_balls(o_balls, o_template, o_at)
        move_balls(h_balls, h_template, h_at)
        paint_glow(frame)
        paint_labels(frame, o_at)
        wire.points = positions[frame].astype(float)
        if not np.array_equal(bonds[frame], state["bonds"]):  # a reaction happened
            wire.lines = np.column_stack(
                [np.full(len(bonds[frame]), 2), bonds[frame]]).ravel()
            state["bonds"] = bonds[frame]

        counts = np.bincount(bonds[frame, :, 0] // 3, minlength=o_mask.sum())
        event, _ = active_event(shots, frame)
        text = (f"frame {frame + 1}/{n_frames}    "
                f"H3O+: {int((counts == 3).sum())}   OH-: {int((counts == 1).sum())}\n"
                f"{TITLES[event[1]] if event else ''}")
        hud.SetText(2, text)  # 2 = the upper-left corner
        if state["auto_cam"]:
            place_camera(frame)

    def tick(_step=None):
        if state["playing"]:
            state["frame"] = (state["frame"] + 1) % n_frames
        show_frame(state["frame"])
        pl.render()

    if args.snapshot is not None:  # picture mode (used for testing)
        for frame in args.snapshot:
            show_frame(frame)
            pl.render()
            pl.screenshot(f"snapshot_{frame:04d}.png")
            print(f"📷 snapshot_{frame:04d}.png")
        return

    pl.add_key_event("space", lambda: state.update(playing=not state["playing"]))
    pl.add_key_event("a", lambda: state.update(auto_cam=not state["auto_cam"]))
    pl.add_key_event("r", lambda: state.update(frame=0))
    show_frame(0)
    pl.add_timer_event(max_steps=10 ** 9, duration=33, callback=tick)
    pl.show()


if __name__ == "__main__":
    main()
