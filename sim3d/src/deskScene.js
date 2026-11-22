import * as THREE from 'three';
import {
  createBrushedMetalTexture,
  createFeltMatTextures,
  createNotebookPageTexture,
  createWalnutWoodTextures,
} from './textures.js';

/**
 * Create a gracefully arched open book page geometry (realistic spine curve!).
 */
function createCurvedPageGeo(w, h, side = 'left') {
  const segX = 24;
  const segY = 8;
  const geo = new THREE.PlaneGeometry(w, h, segX, segY);
  const pos = geo.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i); // -w/2 .. +w/2
    // u = 0 at spine, 1 at outer page edge
    const u = side === 'left' ? (w / 2 - x) / w : (x + w / 2) / w;
    // Realistic open-hardcover book arch rising out of spine then settling gently
    const arch = Math.sin(u * Math.PI) * 0.055 + Math.pow(u, 0.55) * 0.025;
    pos.setZ(i, arch);
  }
  geo.computeVertexNormals();
  geo.rotateX(-Math.PI / 2);
  return geo;
}

/**
 * Photorealistic Interactive Study Desk Scene:
 * - Architectural Walnut Desk + Charcoal Wool Felt Desk Mat
 * - Draggable Hardcover Open Study Notebook (crisp curved pages)
 * - Click-to-Point Virtual Finger / Stylus Target on the Notebook
 * - Studio Desk Props (Aluminum Laptop, Ceramic Espresso Cup, Brass Pen)
 * - Draggable Student Avatar + Toggleable 2nd Guest/Jury Avatar (Multi-Face Social Gaze)
 * - 3 Lighting Moods: NIGHT (Cozy Lamp Showcase) / GOLDEN (Sunset) / DAY (Apple Studio)
 */
export class DeskScene {
  constructor(scene) {
    this.scene = scene;
    this.draggables = [];

    const wood = createWalnutWoodTextures();
    const felt = createFeltMatTextures();
    const metalBump = createBrushedMetalTexture();

    // =========================================================================
    // 1. ARCHITECTURAL WALNUT DESK & STUDIO BACKDROP
    // =========================================================================
    const deskMat = new THREE.MeshStandardMaterial({
      map: wood.map,
      bumpMap: wood.bumpMap,
      bumpScale: 0.015,
      roughness: 0.44,
      metalness: 0.04,
    });
    const deskTop = new THREE.Mesh(new THREE.BoxGeometry(16, 0.36, 9.6), deskMat);
    deskTop.position.set(0, -0.18, 0.8);
    deskTop.receiveShadow = true;
    scene.add(deskTop);
    this.deskSurface = deskTop;

    // Charcoal wool felt desk pad
    const feltPad = new THREE.Mesh(
      new THREE.BoxGeometry(8.8, 0.026, 4.9),
      new THREE.MeshStandardMaterial({
        map: felt.map,
        bumpMap: felt.bumpMap,
        bumpScale: 0.008,
        roughness: 0.92,
        metalness: 0.02,
      })
    );
    feltPad.position.set(0, 0.013, 1.35);
    feltPad.receiveShadow = true;
    scene.add(feltPad);

    // Subtle matte architectural rear wall & warm/cool backdrop
    const wallMat = new THREE.MeshStandardMaterial({
      color: 0x161922,
      roughness: 0.92,
      metalness: 0.02,
    });
    const rearWall = new THREE.Mesh(new THREE.PlaneGeometry(24, 14), wallMat);
    rearWall.position.set(0, 5.2, -4.1);
    rearWall.receiveShadow = true;
    scene.add(rearWall);
    this.rearWallMat = wallMat;

    // =========================================================================
    // 2. DRAGGABLE OPEN HARDCOVER STUDY NOTEBOOK (Interactive Target!)
    // =========================================================================
    this.notebook = new THREE.Group();
    this.notebook.position.set(-0.65, 0.028, 2.05);
    this.notebook.rotation.y = 0.08;
    this.notebook.userData = { dragKind: 'notebook' };
    scene.add(this.notebook);

    const coverMat = new THREE.MeshStandardMaterial({
      color: 0x1b2434,
      roughness: 0.68,
      metalness: 0.08,
    });
    const coverBase = new THREE.Mesh(new THREE.BoxGeometry(3.26, 0.036, 2.18), coverMat);
    coverBase.position.y = 0.018;
    coverBase.castShadow = true;
    coverBase.receiveShadow = true;
    this.notebook.add(coverBase);

    // Paper stack edge blocks underneath curved top pages
    const stackMat = new THREE.MeshStandardMaterial({ color: 0xe6dfd1, roughness: 0.85 });
    const stackL = new THREE.Mesh(new THREE.BoxGeometry(1.52, 0.06, 2.05), stackMat);
    stackL.position.set(-0.78, 0.06, 0);
    stackL.castShadow = true;
    stackL.receiveShadow = true;
    this.notebook.add(stackL);

    const stackR = new THREE.Mesh(new THREE.BoxGeometry(1.52, 0.06, 2.05), stackMat);
    stackR.position.set(0.78, 0.06, 0);
    stackR.castShadow = true;
    stackR.receiveShadow = true;
    this.notebook.add(stackR);

    // Curved Left & Right Pages with crisp high-DPI economics/physics typography
    const leftPageMat = new THREE.MeshStandardMaterial({
      map: createNotebookPageTexture('left'),
      roughness: 0.84,
      metalness: 0.0,
    });
    const rightPageMat = new THREE.MeshStandardMaterial({
      map: createNotebookPageTexture('right'),
      roughness: 0.84,
      metalness: 0.0,
    });

    const pageL = new THREE.Mesh(createCurvedPageGeo(1.54, 2.04, 'left'), leftPageMat);
    pageL.position.set(-0.77, 0.092, 0);
    pageL.receiveShadow = true;
    pageL.userData = { dragKind: 'notebook', isPage: true };
    this.notebook.add(pageL);

    const pageR = new THREE.Mesh(createCurvedPageGeo(1.54, 2.04, 'right'), rightPageMat);
    pageR.position.set(0.77, 0.092, 0);
    pageR.receiveShadow = true;
    pageR.userData = { dragKind: 'notebook', isPage: true };
    this.notebook.add(pageR);

    // Bookmark ribbon peeking from top spine
    const ribbon = new THREE.Mesh(
      new THREE.BoxGeometry(0.05, 0.008, 0.38),
      new THREE.MeshStandardMaterial({ color: 0xc05621, roughness: 0.7 })
    );
    ribbon.position.set(0.02, 0.08, 1.15);
    ribbon.rotation.y = 0.25;
    this.notebook.add(ribbon);

    this.notebookHitMeshes = [coverBase, pageL, pageR];
    this.draggables.push(...this.notebookHitMeshes);

    // =========================================================================
    // 3. INTERACTIVE FINGER-POINTING TARGET (Click anywhere on desk/notebook!)
    // =========================================================================
    this.pointerGroup = new THREE.Group();
    this.pointerGroup.position.set(-0.15, 0.145, 2.08);
    this.pointerGroup.visible = false;
    scene.add(this.pointerGroup);

    // Translucent focus reticle ring on paper (visualizes Apple Vision 11ms Finger Macro!)
    this.pointerRingMat = new THREE.MeshBasicMaterial({
      color: 0xffd088,
      transparent: true,
      opacity: 0.85,
      side: THREE.DoubleSide,
    });
    const ring = new THREE.Mesh(new THREE.RingGeometry(0.12, 0.155, 36), this.pointerRingMat);
    ring.rotation.x = -Math.PI / 2;
    this.pointerGroup.add(ring);

    // Stylus / fingertip sculptural pin pointing at the spot
    const tipMat = new THREE.MeshPhysicalMaterial({
      color: 0xffe0b2,
      emissive: 0xffb85c,
      emissiveIntensity: 0.45,
      roughness: 0.25,
      metalness: 0.1,
      clearcoat: 0.8,
    });
    const stylus = new THREE.Mesh(new THREE.CapsuleGeometry(0.042, 0.55, 8, 16), tipMat);
    stylus.position.set(0.18, 0.34, 0.26);
    stylus.rotation.set(0.62, 0, 0.45);
    stylus.castShadow = true;
    this.pointerGroup.add(stylus);

    // =========================================================================
    // 4. REALISTIC DESK PROPS (Laptop, Ceramic Mug, Brass Pen)
    // =========================================================================
    const aluMat = new THREE.MeshStandardMaterial({
      color: 0xa6aeb8,
      roughness: 0.26,
      metalness: 0.88,
      bumpMap: metalBump,
      bumpScale: 0.002,
    });

    // Minimalist aluminum laptop resting in back-left
    const laptop = new THREE.Group();
    laptop.position.set(-3.35, 0.028, -0.35);
    laptop.rotation.y = 0.46;
    scene.add(laptop);

    const lapBase = new THREE.Mesh(new THREE.BoxGeometry(2.8, 0.075, 1.95), aluMat);
    lapBase.position.y = 0.038;
    lapBase.castShadow = true;
    lapBase.receiveShadow = true;
    laptop.add(lapBase);

    const lapKb = new THREE.Mesh(
      new THREE.BoxGeometry(2.35, 0.012, 0.92),
      new THREE.MeshStandardMaterial({ color: 0x161920, roughness: 0.6 })
    );
    lapKb.position.set(0, 0.076, -0.15);
    laptop.add(lapKb);

    const lapScreenGroup = new THREE.Group();
    lapScreenGroup.position.set(0, 0.075, -0.94);
    lapScreenGroup.rotation.x = -0.34;
    laptop.add(lapScreenGroup);

    const lapLid = new THREE.Mesh(new THREE.BoxGeometry(2.78, 1.82, 0.042), aluMat);
    lapLid.position.set(0, 0.91, 0);
    lapLid.castShadow = true;
    lapScreenGroup.add(lapLid);

    const lapDisplay = new THREE.Mesh(
      new THREE.PlaneGeometry(2.6, 1.64),
      new THREE.MeshStandardMaterial({
        color: 0x111927,
        emissive: 0x1e3354,
        emissiveIntensity: 0.55,
        roughness: 0.12,
      })
    );
    lapDisplay.position.set(0, 0.91, 0.023);
    lapScreenGroup.add(lapDisplay);

    // Ceramic matte espresso mug on cork coaster (back-right)
    const mugGroup = new THREE.Group();
    mugGroup.position.set(2.85, 0.028, 0.55);
    scene.add(mugGroup);

    const coaster = new THREE.Mesh(
      new THREE.CylinderGeometry(0.46, 0.47, 0.03, 32),
      new THREE.MeshStandardMaterial({ color: 0x5c402d, roughness: 0.85 })
    );
    coaster.position.y = 0.015;
    coaster.receiveShadow = true;
    mugGroup.add(coaster);

    const mugMat = new THREE.MeshStandardMaterial({ color: 0xd9d2c5, roughness: 0.38, metalness: 0.04 });
    const mugBody = new THREE.Mesh(new THREE.CylinderGeometry(0.34, 0.31, 0.72, 32), mugMat);
    mugBody.position.y = 0.38;
    mugBody.castShadow = true;
    mugBody.receiveShadow = true;
    mugGroup.add(mugBody);

    const coffeeTop = new THREE.Mesh(
      new THREE.CircleGeometry(0.31, 28),
      new THREE.MeshPhysicalMaterial({ color: 0x26150a, roughness: 0.08, clearcoat: 1.0 })
    );
    coffeeTop.rotation.x = -Math.PI / 2;
    coffeeTop.position.y = 0.72;
    mugGroup.add(coffeeTop);

    const mugHandle = new THREE.Mesh(new THREE.TorusGeometry(0.19, 0.04, 14, 28, Math.PI), mugMat);
    mugHandle.position.set(0.32, 0.4, 0);
    mugHandle.rotation.z = -Math.PI / 2;
    mugHandle.castShadow = true;
    mugGroup.add(mugHandle);

    // Brushed brass fountain pen resting next to notebook
    const penMat = new THREE.MeshStandardMaterial({ color: 0xc9a462, roughness: 0.25, metalness: 0.88 });
    const pen = new THREE.Mesh(new THREE.CylinderGeometry(0.036, 0.032, 1.32, 20), penMat);
    pen.rotation.z = Math.PI / 2;
    pen.rotation.y = -0.32;
    pen.position.set(1.52, 0.06, 2.25);
    pen.castShadow = true;
    scene.add(pen);

    // =========================================================================
    // 5. INTERACTIVE GAZE AVATARS (Student + Optional Guest/Jury for Multi-Face)
    // =========================================================================
    this.studentMarker = this._createAvatarMarker(0x60a5fa, 'ZACH');
    this.studentMarker.position.set(0.0, 3.65, 4.35);
    this.studentMarker.userData = { dragKind: 'student' };
    scene.add(this.studentMarker);
    this.draggables.push(this.studentMarker.children[0]);

    this.guestMarker = this._createAvatarMarker(0x34d399, 'INVITÉ');
    this.guestMarker.position.set(2.45, 3.55, 3.65);
    this.guestMarker.visible = false;
    this.guestMarker.userData = { dragKind: 'guest' };
    scene.add(this.guestMarker);
    this.draggables.push(this.guestMarker.children[0]);

    // =========================================================================
    // 6. STUDIO THREE-POINT LIGHTING RIG (DAY / GOLDEN / NIGHT MOODS)
    // =========================================================================
    this.hemiLight = new THREE.HemisphereLight(0x9fb8e8, 0x261d17, 0.45);
    scene.add(this.hemiLight);

    this.keyLight = new THREE.DirectionalLight(0xfff1dc, 1.25);
    this.keyLight.position.set(-4.5, 7.5, 4.5);
    this.keyLight.castShadow = true;
    this.keyLight.shadow.mapSize.set(2048, 2048);
    this.keyLight.shadow.camera.near = 1;
    this.keyLight.shadow.camera.far = 22;
    this.keyLight.shadow.camera.left = -7;
    this.keyLight.shadow.camera.right = 7;
    this.keyLight.shadow.camera.top = 7;
    this.keyLight.shadow.camera.bottom = -7;
    this.keyLight.shadow.bias = -0.0005;
    scene.add(this.keyLight);

    this.rimLight = new THREE.DirectionalLight(0x7ca7ec, 0.9);
    this.rimLight.position.set(5.5, 4.8, -5.5);
    scene.add(this.rimLight);

    this.setMood('night');
  }

  _createAvatarMarker(colorHex, label) {
    const g = new THREE.Group();
    const orb = new THREE.Mesh(
      new THREE.SphereGeometry(0.24, 24, 24),
      new THREE.MeshPhysicalMaterial({
        color: colorHex,
        emissive: colorHex,
        emissiveIntensity: 0.45,
        roughness: 0.18,
        metalness: 0.1,
        transparent: true,
        opacity: 0.72,
      })
    );
    orb.userData = { dragKind: label === 'ZACH' ? 'student' : 'guest' };
    g.add(orb);

    const halo = new THREE.Mesh(
      new THREE.TorusGeometry(0.32, 0.012, 12, 36),
      new THREE.MeshBasicMaterial({ color: colorHex, transparent: true, opacity: 0.5 })
    );
    halo.rotation.x = Math.PI / 2;
    g.add(halo);
    return g;
  }

  setMood(mood) {
    if (mood === 'day') {
      this.scene.background = new THREE.Color(0x1f242e);
      this.scene.fog = new THREE.FogExp2(0x1f242e, 0.032);
      this.rearWallMat.color.set(0x2a313f);
      this.hemiLight.color.set(0xe4eeff);
      this.hemiLight.groundColor.set(0x5b4636);
      this.hemiLight.intensity = 1.05;
      this.keyLight.color.set(0xfffaf2);
      this.keyLight.intensity = 2.1;
      this.keyLight.position.set(-4.5, 8.2, 4.8);
      this.rimLight.color.set(0xa4c8ff);
      this.rimLight.intensity = 0.85;
    } else if (mood === 'golden') {
      this.scene.background = new THREE.Color(0x171418);
      this.scene.fog = new THREE.FogExp2(0x171418, 0.038);
      this.rearWallMat.color.set(0x211a1d);
      this.hemiLight.color.set(0xffd2a0);
      this.hemiLight.groundColor.set(0x2a1c15);
      this.hemiLight.intensity = 0.6;
      this.keyLight.color.set(0xffb46b);
      this.keyLight.intensity = 1.65;
      this.keyLight.position.set(-7.5, 4.2, 3.8);
      this.rimLight.color.set(0x6d93d6);
      this.rimLight.intensity = 0.95;
    } else {
      // Cozy Night Studio — showcases the warm HiLight volumetric desk lamp dramatically!
      this.scene.background = new THREE.Color(0x090b11);
      this.scene.fog = new THREE.FogExp2(0x090b11, 0.044);
      this.rearWallMat.color.set(0x11141d);
      this.hemiLight.color.set(0x6c8cc2);
      this.hemiLight.groundColor.set(0x16110e);
      this.hemiLight.intensity = 0.34;
      this.keyLight.color.set(0xc9ddff);
      this.keyLight.intensity = 0.52;
      this.keyLight.position.set(-4.5, 7.5, 4.5);
      this.rimLight.color.set(0x6897e8);
      this.rimLight.intensity = 0.95;
    }
  }

  getNotebookCenter() {
    return new THREE.Vector3(this.notebook.position.x, 0.14, this.notebook.position.z);
  }

  getActiveDeskTarget() {
    if (this.pointerGroup.visible) {
      return this.pointerGroup.position.clone();
    }
    return this.getNotebookCenter();
  }
}
