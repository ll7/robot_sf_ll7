// Optional restrained 2.5D presentation view (issue #9369).
//
// A narrow view component over an already-exported scene: `top-down`
// reproduces the existing overhead camera, `isometric` adds wall volumes
// extruded from the exact source obstacle polygons (bevels disabled so floor
// footprints are unchanged) under a fixed elevated orthographic camera.
// Wall height is illustrative and disclosed; simulation geometry stays 2D.

export const ISOMETRIC_ELEVATION_DEG = 50.0;
export const ISOMETRIC_AZIMUTH_DEG = 45.0;

export function viewPreset(payload) {
  const preset = payload?.view?.preset;
  return preset === "isometric" ? "isometric" : "top-down";
}

export function wallHeight(payload) {
  const height = payload?.view?.walls?.height_m;
  if (typeof height === "number" && Number.isFinite(height) && height >= 0) {
    return height;
  }
  return 1.2;
}

export function buildWallVolumes(THREE, map, height) {
  // Extrude each obstacle polygon straight up; bevels stay disabled so the
  // floor footprint equals the source vertices exactly.
  const group = new THREE.Group();
  (map.obstacles || []).forEach((obstacle) => {
    const shape = new THREE.Shape(
      (obstacle.vertices || []).map(([x, y]) => new THREE.Vector2(x, y))
    );
    const geometry = new THREE.ExtrudeGeometry(shape, {
      depth: height,
      bevelEnabled: false,
    });
    geometry.rotateX(-Math.PI / 2);
    const mesh = new THREE.Mesh(
      geometry,
      new THREE.MeshLambertMaterial({ color: 0x8b98ab, transparent: true, opacity: 0.92 })
    );
    group.add(mesh);
  });
  return group;
}

export function poseIsometricCamera(camera, centerX, centerZ, span) {
  const elevation = (ISOMETRIC_ELEVATION_DEG * Math.PI) / 180;
  const azimuth = (ISOMETRIC_AZIMUTH_DEG * Math.PI) / 180;
  const distance = Math.max(span, 1) * 2.1;
  camera.position.set(
    centerX + distance * Math.cos(elevation) * Math.cos(azimuth),
    distance * Math.sin(elevation),
    centerZ + distance * Math.cos(elevation) * Math.sin(azimuth)
  );
  camera.up.set(0, 1, 0);
  camera.lookAt(centerX, 0, centerZ);
  camera.updateProjectionMatrix();
}

export function presentationDisclosures(payload) {
  const view = payload?.view;
  if (!view) return [];
  return Array.isArray(view.disclosures) ? view.disclosures : [];
}
