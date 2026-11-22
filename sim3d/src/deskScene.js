import * as THREE from 'three';
import {
  createBrushedMetalTexture,
  createFeltMatTextures,
  createMacBookKeyboardTexture,
  createMacBookScreenTexture,
  createNotebookPageTexture,
  createOakWoodTextures,
} from './textures.js';
import { createRoundedBoxGeo, createRoundedPlaneGeo } from './robotRig.js';

/**
 * Gracefully arched open book page geometry (realistic hardcover spine curve).
 */
function createCurvedPageGeo(w, h, side = 'left') {
  const segX = 24;
  const segY = 8;
  const geo = new THREE.PlaneGeometry(w, h, segX, segY);
  const pos = geo.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i);
    const u = side === 'left' ? (w / 2 - x) / w : (x + w / 2) / w;
    const arch = Math.sin(u * Math.PI) * 0.055 + Math.pow(u, 0.55) * 0.025;
    pos.setZ(i, arch);
  }
  geo.computeVertexNormals();
  geo.rotateX(-Math.PI / 2);
  return geo;
}

/**
 * Sculpted curved botanical leaf geometry (natural arch + midrib cup).
 */
function createBotanicalLeafGeo(len = 0.72, width = 0.34) {
  const shape = new THREE.Shape();
  shape.moveTo(0, 0);
  shape.bezierCurveTo(width * 0.65, len * 0.2, width * 0.58, len * 0.72, 0, len);
  shape.bezierCurveTo(-width * 0.58, len * 0.72, -width * 0.65, len * 0.2, 0, 0);

  const geo = new THREE.ShapeGeometry(shape, 14);
  const pos = geo.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i);
    const y = pos.getY(i);
    const t = Math.max(0, Math.min(1, y / len));
    // Gentle downward tip droop + subtle V-fold along midrib
    const droop = -Math.pow(t, 2.0) * 0.19;
    const cup = Math.abs(x) * 0.14 * Math.sin(t * Math.PI);
    pos.setZ(i, droop + cup);
  }
  geo.computeVertexNormals();
  return geo;
}

/**
 * Minimalist framed Bauhaus art canvas texture for the wall shelf.
 */
function createWallArtTexture() {
  const w = 512;
  const h = 640;
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  const ctx = c.getContext('2d');

  ctx.fillStyle = '#f5f1e8';
  ctx.fillRect(0, 0, w, h);

  // Warm terracotta arch
  ctx.fillStyle = '#c86b4a';
  ctx.beginPath();
  ctx.arc(256, 360, 145, Math.PI, 0, false);
  ctx.lineTo(401, 520);
  ctx.lineTo(111, 520);
  ctx.closePath();
  ctx.fill();

  // Deep slate sun circle
  ctx.fillStyle = '#243040';
  ctx.beginPath();
  ctx.arc(312, 198, 66, 0, Math.PI * 2);
  ctx.fill();

  // Minimalist fine balance line
  ctx.strokeStyle = '#1e2530';
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.moveTo(86, 522);
  ctx.lineTo(426, 522);
  ctx.stroke();

  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

/**
 * Bright Sunlit Architectural Study Room + Photorealistic MacBook Pro 16" + Desk Props:
 * - Warm off-white plaster walls, tall sunlit window on the left with mullions & sun shaft
 * - Light Scandinavian Oak desk + warm architectural stone-gray felt desk mat
 * - Photorealistic CNC MacBook Pro 16" (backlit keyboard + live macOS Osmo Studio screen)
 * - Floating oak wall shelf (art books, fluted ceramic vase, framed minimalist print)
 * - Potted botanical desk plant (Apple ELEGNT nod!)
 * - Hollowed stoneware espresso mug + sculpted brushed pen + draggable open study notebook
 * - Default mood: DAY (bright sunlit room) + GOLDEN (sunset) + NIGHT (cozy lamp showcase)
 */
export class DeskScene {
  constructor(scene) {
    this.scene = scene;
    this.draggables = [];

    const oak = createOakWoodTextures();
    const felt = createFeltMatTextures();
    const metalBump = createBrushedMetalTexture();

    // =========================================================================
    // 1. BRIGHT SUNLIT ARCHITECTURAL ROOM (Walls, Window, Floor, Shelf)
    // =========================================================================
    this.wallMat = new THREE.MeshStandardMaterial({
      color: 0xede9e1,
      roughness: 0.92,
      metalness: 0.0,
    });
    const trimMat = new THREE.MeshStandardMaterial({
      color: 0xf7f5f0,
      roughness: 0.65,
      metalness: 0.02,
    });
    const mullionMat = new THREE.MeshStandardMaterial({
      color: 0x2c3038,
      roughness: 0.42,
      metalness: 0.45,
    });

    // Rear architectural plaster wall
    const rearWall = new THREE.Mesh(new THREE.PlaneGeometry(24, 15), this.wallMat);
    rearWall.position.set(0.5, 3.2, -3.9);
    rearWall.receiveShadow = true;
    scene.add(rearWall);

    // Right return wall (encloses the room cleanly when orbiting!)
    const rightWall = new THREE.Mesh(new THREE.PlaneGeometry(18, 15), this.wallMat);
    rightWall.rotation.y = -Math.PI / 2;
    rightWall.position.set(8.6, 3.2, 3.2);
    rightWall.receiveShadow = true;
    scene.add(rightWall);

    // Ceiling plane (bounces bright soft architectural tone)
    const ceiling = new THREE.Mesh(new THREE.PlaneGeometry(24, 18), this.wallMat);
    ceiling.rotation.x = Math.PI / 2;
    ceiling.position.set(0.5, 8.8, 3.0);
    scene.add(ceiling);

    // Warm oak room floor underneath desk
    const floorMat = new THREE.MeshStandardMaterial({
      map: oak.map,
      roughness: 0.62,
      metalness: 0.02,
      color: 0xbfa382,
    });
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(24, 18), floorMat);
    floor.rotation.x = -Math.PI / 2;
    floor.position.set(0.5, -4.35, 3.0);
    floor.receiveShadow = true;
    scene.add(floor);

    // Baseboard trim along rear wall
    const baseboard = new THREE.Mesh(new THREE.BoxGeometry(22, 0.55, 0.1), trimMat);
    baseboard.position.set(0.5, -4.08, -3.85);
    scene.add(baseboard);

    // --- LEFT ARCHITECTURAL WALL WITH TALL SUNLIT WINDOW ---
    const winX = -6.85;
    // Wall sections framing window opening (z: -1.8 .. +3.2, y: 0.25 .. 6.4)
    const leftBackCol = new THREE.Mesh(new THREE.BoxGeometry(0.45, 13.5, 2.3), this.wallMat);
    leftBackCol.position.set(winX, 2.4, -2.95);
    leftBackCol.receiveShadow = true;
    scene.add(leftBackCol);

    const leftFrontCol = new THREE.Mesh(new THREE.BoxGeometry(0.45, 13.5, 6.2), this.wallMat);
    leftFrontCol.position.set(winX, 2.4, 6.3);
    leftFrontCol.receiveShadow = true;
    scene.add(leftFrontCol);

    const leftBelowWin = new THREE.Mesh(new THREE.BoxGeometry(0.45, 4.7, 5.2), this.wallMat);
    leftBelowWin.position.set(winX, -2.05, 0.7);
    leftBelowWin.receiveShadow = true;
    scene.add(leftBelowWin);

    const leftAboveWin = new THREE.Mesh(new THREE.BoxGeometry(0.45, 2.8, 5.2), this.wallMat);
    leftAboveWin.position.set(winX, 7.65, 0.7);
    scene.add(leftAboveWin);

    // Window sill + slim architectural charcoal mullions
    const sill = new THREE.Mesh(new THREE.BoxGeometry(0.66, 0.14, 5.25), trimMat);
    sill.position.set(winX + 0.08, 0.28, 0.7);
    sill.receiveShadow = true;
    scene.add(sill);

    for (const mz of [-1.75, 0.7, 3.15]) {
      const vBar = new THREE.Mesh(new THREE.BoxGeometry(0.12, 6.1, 0.09), mullionMat);
      vBar.position.set(winX - 0.04, 3.3, mz);
      scene.add(vBar);
    }
    for (const my of [0.35, 3.35, 6.25]) {
      const hBar = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.09, 5.0), mullionMat);
      hBar.position.set(winX - 0.04, my, 0.7);
      scene.add(hBar);
    }

    // Luminous sunlit sky panel outside the window
    this.windowSkyMat = new THREE.MeshBasicMaterial({ color: 0xf4f8ff });
    const skyPanel = new THREE.Mesh(new THREE.PlaneGeometry(5.6, 6.6), this.windowSkyMat);
    skyPanel.rotation.y = Math.PI / 2;
    skyPanel.position.set(winX - 0.26, 3.3, 0.7);
    scene.add(skyPanel);

    // Soft feathered diagonal sunlight shaft entering from the left window (zero hard edges!)
    const shaftCanvas = document.createElement('canvas');
    shaftCanvas.width = 256;
    shaftCanvas.height = 256;
    const sctx = shaftCanvas.getContext('2d');
    const sg = sctx.createRadialGradient(72, 128, 12, 128, 128, 124);
    sg.addColorStop(0.0, 'rgba(255, 248, 232, 0.9)');
    sg.addColorStop(0.45, 'rgba(255, 246, 224, 0.35)');
    sg.addColorStop(1.0, 'rgba(255, 246, 224, 0.0)');
    sctx.fillStyle = sg;
    sctx.fillRect(0, 0, 256, 256);
    const shaftTex = new THREE.CanvasTexture(shaftCanvas);

    this.sunShaftMat = new THREE.MeshBasicMaterial({
      map: shaftTex,
      color: 0xfff7e6,
      transparent: true,
      opacity: 0.14,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      side: THREE.DoubleSide,
    });
    const sunShaft = new THREE.Mesh(new THREE.PlaneGeometry(7.2, 4.8), this.sunShaftMat);
    sunShaft.position.set(-2.9, 2.35, 0.45);
    sunShaft.rotation.set(0.12, 0.26, -0.52);
    scene.add(sunShaft);

    // --- FLOATING LIGHT-OAK WALL SHELF + BOOKS + VASE + FRAMED ART ---
    const oakMat = new THREE.MeshStandardMaterial({
      map: oak.map,
      bumpMap: oak.bumpMap,
      bumpScale: 0.012,
      roughness: 0.48,
      metalness: 0.02,
    });

    const shelfGroup = new THREE.Group();
    shelfGroup.position.set(1.6, 3.9, -3.45);
    scene.add(shelfGroup);

    const shelfPlank = new THREE.Mesh(createRoundedBoxGeo(5.6, 0.16, 0.88, 0.03, 0.01), oakMat);
    shelfPlank.castShadow = true;
    shelfPlank.receiveShadow = true;
    shelfGroup.add(shelfPlank);

    // Minimalist framed art print leaning on shelf left
    const artGroup = new THREE.Group();
    artGroup.position.set(-1.55, 1.02, -0.14);
    artGroup.rotation.x = -0.11;
    shelfGroup.add(artGroup);

    const artFrame = new THREE.Mesh(createRoundedBoxGeo(1.48, 1.88, 0.07, 0.02, 0.008), oakMat);
    artFrame.castShadow = true;
    artGroup.add(artFrame);

    const artCanvas = new THREE.Mesh(
      new THREE.PlaneGeometry(1.32, 1.72),
      new THREE.MeshStandardMaterial({ map: createWallArtTexture(), roughness: 0.85 })
    );
    artCanvas.position.z = 0.038;
    artGroup.add(artCanvas);

    // Stack of 3 architectural hardcover books
    const bookPalette = [
      { w: 1.12, h: 0.16, d: 0.72, col: 0x3b4d5e, y: 0.16, rot: 0.05 },
      { w: 1.02, h: 0.14, d: 0.68, col: 0xc26d4f, y: 0.31, rot: -0.04 },
      { w: 0.94, h: 0.12, d: 0.62, col: 0xe6dfd2, y: 0.44, rot: 0.08 },
    ];
    bookPalette.forEach((b) => {
      const bm = new THREE.Mesh(
        createRoundedBoxGeo(b.w, b.h, b.d, 0.02, 0.006),
        new THREE.MeshStandardMaterial({ color: b.col, roughness: 0.74 })
      );
      bm.position.set(0.35, b.y, 0.02);
      bm.rotation.y = b.rot;
      bm.castShadow = true;
      bm.receiveShadow = true;
      shelfGroup.add(bm);
    });

    // Sculptural matte ceramic vase (LatheGeometry) on shelf right
    const vasePts = [
      new THREE.Vector2(0.0, 0.0),
      new THREE.Vector2(0.22, 0.0),
      new THREE.Vector2(0.32, 0.25),
      new THREE.Vector2(0.3, 0.58),
      new THREE.Vector2(0.14, 0.82),
      new THREE.Vector2(0.16, 1.06),
      new THREE.Vector2(0.13, 1.06),
      new THREE.Vector2(0.11, 0.8),
    ];
    const vase = new THREE.Mesh(
      new THREE.LatheGeometry(vasePts, 36),
      new THREE.MeshStandardMaterial({ color: 0xf0ece4, roughness: 0.55, metalness: 0.02 })
    );
    vase.position.set(1.78, 0.08, 0.02);
    vase.castShadow = true;
    vase.receiveShadow = true;
    shelfGroup.add(vase);

    // =========================================================================
    // 2. SCANDINAVIAN LIGHT-OAK DESK + WARM STONE-GRAY FELT MAT
    // =========================================================================
    const deskTop = new THREE.Mesh(createRoundedBoxGeo(13.6, 0.34, 7.6, 0.09, 0.024, 16), oakMat);
    deskTop.position.set(0.0, -0.17, 0.45);
    deskTop.castShadow = true;
    deskTop.receiveShadow = true;
    scene.add(deskTop);
    this.deskSurface = deskTop;

    // Minimalist architectural matte-black steel desk legs under the oak top
    const legMat = new THREE.MeshStandardMaterial({ color: 0x262a31, roughness: 0.48, metalness: 0.65 });
    for (const lx of [-5.8, 5.8]) {
      for (const lz of [-2.4, 3.2]) {
        const dLeg = new THREE.Mesh(new THREE.CylinderGeometry(0.11, 0.085, 4.1, 20), legMat);
        dLeg.position.set(lx, -2.35, lz);
        scene.add(dLeg);
      }
    }

    // Warm architectural stone-gray wool felt desk mat (with clean rounded corners!)
    const feltMat = new THREE.MeshStandardMaterial({
      map: felt.map,
      bumpMap: felt.bumpMap,
      bumpScale: 0.007,
      roughness: 0.94,
      metalness: 0.01,
    });
    const feltPad = new THREE.Mesh(createRoundedBoxGeo(8.9, 0.028, 4.85, 0.14, 0.006, 16), feltMat);
    feltPad.position.set(-0.05, 0.014, 1.15);
    feltPad.receiveShadow = true;
    scene.add(feltPad);

    // =========================================================================
    // 3. DRAGGABLE OPEN HARDCOVER STUDY NOTEBOOK
    // =========================================================================
    this.notebook = new THREE.Group();
    this.notebook.position.set(-0.58, 0.03, 2.02);
    this.notebook.rotation.y = 0.07;
    this.notebook.userData = { dragKind: 'notebook' };
    scene.add(this.notebook);

    const coverMat = new THREE.MeshStandardMaterial({
      color: 0x1e2c42,
      roughness: 0.72,
      metalness: 0.06,
    });
    const coverBase = new THREE.Mesh(createRoundedBoxGeo(3.26, 0.038, 2.18, 0.05, 0.006, 14), coverMat);
    coverBase.position.y = 0.019;
    coverBase.castShadow = true;
    coverBase.receiveShadow = true;
    this.notebook.add(coverBase);

    const stackMat = new THREE.MeshStandardMaterial({ color: 0xe9e2d4, roughness: 0.86 });
    const stackL = new THREE.Mesh(new THREE.BoxGeometry(1.52, 0.062, 2.05), stackMat);
    stackL.position.set(-0.78, 0.062, 0);
    stackL.castShadow = true;
    stackL.receiveShadow = true;
    this.notebook.add(stackL);

    const stackR = new THREE.Mesh(new THREE.BoxGeometry(1.52, 0.062, 2.05), stackMat);
    stackR.position.set(0.78, 0.062, 0);
    stackR.castShadow = true;
    stackR.receiveShadow = true;
    this.notebook.add(stackR);

    const leftPageMat = new THREE.MeshStandardMaterial({
      map: createNotebookPageTexture('left'),
      roughness: 0.85,
      metalness: 0.0,
    });
    const rightPageMat = new THREE.MeshStandardMaterial({
      map: createNotebookPageTexture('right'),
      roughness: 0.85,
      metalness: 0.0,
    });

    const pageL = new THREE.Mesh(createCurvedPageGeo(1.54, 2.04, 'left'), leftPageMat);
    pageL.position.set(-0.77, 0.094, 0);
    pageL.receiveShadow = true;
    pageL.userData = { dragKind: 'notebook', isPage: true };
    this.notebook.add(pageL);

    const pageR = new THREE.Mesh(createCurvedPageGeo(1.54, 2.04, 'right'), rightPageMat);
    pageR.position.set(0.77, 0.094, 0);
    pageR.receiveShadow = true;
    pageR.userData = { dragKind: 'notebook', isPage: true };
    this.notebook.add(pageR);

    const ribbon = new THREE.Mesh(
      new THREE.BoxGeometry(0.048, 0.008, 0.4),
      new THREE.MeshStandardMaterial({ color: 0xc25934, roughness: 0.72 })
    );
    ribbon.position.set(0.02, 0.08, 1.15);
    ribbon.rotation.y = 0.24;
    this.notebook.add(ribbon);

    this.notebookHitMeshes = [coverBase, pageL, pageR];
    this.draggables.push(...this.notebookHitMeshes);

    // =========================================================================
    // 4. INTERACTIVE FINGER-POINTING TARGET (Click anywhere on a page!)
    // =========================================================================
    this.pointerGroup = new THREE.Group();
    this.pointerGroup.position.set(-0.15, 0.145, 2.08);
    this.pointerGroup.visible = false;
    scene.add(this.pointerGroup);

    this.pointerRingMat = new THREE.MeshBasicMaterial({
      color: 0xe69f38,
      transparent: true,
      opacity: 0.9,
      side: THREE.DoubleSide,
    });
    const ring = new THREE.Mesh(new THREE.RingGeometry(0.115, 0.15, 36), this.pointerRingMat);
    ring.rotation.x = -Math.PI / 2;
    this.pointerGroup.add(ring);

    const stylusMat = new THREE.MeshPhysicalMaterial({
      color: 0xf2c98a,
      emissive: 0xffa73b,
      emissiveIntensity: 0.28,
      roughness: 0.22,
      metalness: 0.25,
      clearcoat: 0.85,
    });
    const stylus = new THREE.Mesh(new THREE.CapsuleGeometry(0.036, 0.52, 8, 16), stylusMat);
    stylus.position.set(0.16, 0.33, 0.24);
    stylus.rotation.set(0.6, 0, 0.44);
    stylus.castShadow = true;
    this.pointerGroup.add(stylus);

    // =========================================================================
    // 5. PHOTOREALISTIC CNC MACBOOK PRO 16" (Space Silver + Osmo Studio UI)
    // =========================================================================
    const macAluMat = new THREE.MeshPhysicalMaterial({
      color: 0xc8ced6,
      roughness: 0.24,
      metalness: 0.9,
      clearcoat: 0.18,
      clearcoatRoughness: 0.28,
      bumpMap: metalBump,
      bumpScale: 0.0014,
    });
    const macDarkMat = new THREE.MeshStandardMaterial({
      color: 0x111419,
      roughness: 0.45,
      metalness: 0.2,
    });
    const macGlassMat = new THREE.MeshPhysicalMaterial({
      color: 0x05070a,
      roughness: 0.03,
      metalness: 0.08,
      clearcoat: 1.0,
      clearcoatRoughness: 0.02,
    });

    const laptop = new THREE.Group();
    laptop.position.set(-3.12, 0.028, -0.32);
    laptop.rotation.y = 0.44;
    scene.add(laptop);

    // 4 subtle black rubber feet underneath base (realistic contact shadow gap)
    for (const fx of [-1.18, 1.18]) {
      for (const fz of [-0.76, 0.76]) {
        const foot = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.075, 0.016, 16), macDarkMat);
        foot.position.set(fx, 0.008, fz);
        laptop.add(foot);
      }
    }

    // CNC Machined Rounded Aluminum Lower Unibody
    const lapBase = new THREE.Mesh(createRoundedBoxGeo(2.96, 0.088, 2.04, 0.14, 0.02, 24), macAluMat);
    lapBase.position.y = 0.056;
    lapBase.castShadow = true;
    lapBase.receiveShadow = true;
    laptop.add(lapBase);

    // Front thumb-scoop opening recess on center front lip
    const thumbScoop = new THREE.Mesh(createRoundedBoxGeo(0.46, 0.024, 0.05, 0.02, 0.005), macDarkMat);
    thumbScoop.position.set(0, 0.09, 1.005);
    laptop.add(thumbScoop);

    // Side I/O ports (MagSafe 3 + 2x Thunderbolt USB-C + 3.5mm jack on left, HDMI/USB-C/SD on right)
    const addPort = (x, z, w, h) => {
      const p = new THREE.Mesh(new THREE.BoxGeometry(0.012, h, w), macDarkMat);
      p.position.set(x, 0.056, z);
      laptop.add(p);
    };
    addPort(-1.482, -0.64, 0.12, 0.024); // MagSafe 3
    addPort(-1.482, -0.44, 0.075, 0.022); // USB-C 1
    addPort(-1.482, -0.31, 0.075, 0.022); // USB-C 2
    addPort(-1.482, -0.18, 0.026, 0.026); // Audio jack
    addPort(1.482, -0.58, 0.11, 0.024); // HDMI
    addPort(1.482, -0.4, 0.075, 0.022); // USB-C 3
    addPort(1.482, -0.2, 0.14, 0.016); // SDXC slot

    // Full Backlit Keyboard + Speaker Grilles + Force Touch Trackpad Deck
    const kbTex = createMacBookKeyboardTexture();
    const kbDeck = new THREE.Mesh(
      createRoundedPlaneGeo(2.82, 1.9, 0.095, 20),
      new THREE.MeshStandardMaterial({
        map: kbTex,
        roughness: 0.42,
        metalness: 0.45,
      })
    );
    kbDeck.rotation.x = -Math.PI / 2;
    kbDeck.position.set(0, 0.1015, 0.02);
    kbDeck.receiveShadow = true;
    laptop.add(kbDeck);

    // Matte-black cylindrical clutch hinge
    const hinge = new THREE.Mesh(new THREE.CylinderGeometry(0.032, 0.032, 2.28, 20), macDarkMat);
    hinge.rotation.z = Math.PI / 2;
    hinge.position.set(0, 0.102, -0.96);
    laptop.add(hinge);

    // Ultra-thin CNC Rounded Display Lid + Glossy Black Bezel + Liquid Retina Osmo Studio Screen
    const lapScreenGroup = new THREE.Group();
    lapScreenGroup.position.set(0, 0.106, -0.96);
    lapScreenGroup.rotation.x = -0.33; // ~109° ergonomic laptop tilt
    laptop.add(lapScreenGroup);

    const lapLid = new THREE.Mesh(createRoundedBoxGeo(2.94, 1.94, 0.044, 0.13, 0.012, 24), macAluMat);
    lapLid.position.set(0, 0.97, 0);
    lapLid.castShadow = true;
    lapLid.receiveShadow = true;
    lapScreenGroup.add(lapLid);

    // Subtle mirror-chrome emblem disc on back of aluminum lid
    const rearEmblem = new THREE.Mesh(
      new THREE.CircleGeometry(0.14, 32),
      new THREE.MeshPhysicalMaterial({
        color: 0xeef2f7,
        roughness: 0.05,
        metalness: 0.98,
        clearcoat: 1.0,
      })
    );
    rearEmblem.rotation.y = Math.PI;
    rearEmblem.position.set(0, 0.99, -0.0235);
    lapScreenGroup.add(rearEmblem);

    // Edge-to-edge black glass bezel on front of lid
    const lapBezel = new THREE.Mesh(createRoundedPlaneGeo(2.86, 1.86, 0.105, 24), macGlassMat);
    lapBezel.position.set(0, 0.97, 0.023);
    lapScreenGroup.add(lapBezel);

    // Crisp Liquid Retina Display running Osmo Studio macOS UI!
    const screenTex = createMacBookScreenTexture();
    const lapDisplay = new THREE.Mesh(
      createRoundedPlaneGeo(2.74, 1.74, 0.055, 20),
      new THREE.MeshStandardMaterial({
        map: screenTex,
        emissiveMap: screenTex,
        emissive: new THREE.Color(0xffffff),
        emissiveIntensity: 0.58,
        roughness: 0.08,
        metalness: 0.04,
      })
    );
    lapDisplay.position.set(0, 0.975, 0.025);
    lapScreenGroup.add(lapDisplay);

    // =========================================================================
    // 6. LUSH POTTED BOTANICAL DESK PLANT (Apple ELEGNT Inspiration!)
    // =========================================================================
    const plantGroup = new THREE.Group();
    plantGroup.position.set(3.55, 0.0, -1.25);
    scene.add(plantGroup);

    const potPts = [
      new THREE.Vector2(0.0, 0.0),
      new THREE.Vector2(0.46, 0.0),
      new THREE.Vector2(0.58, 0.45),
      new THREE.Vector2(0.56, 0.95),
      new THREE.Vector2(0.5, 0.96),
      new THREE.Vector2(0.48, 0.85),
    ];
    const potMat = new THREE.MeshStandardMaterial({ color: 0xe6ddd0, roughness: 0.62, metalness: 0.02 });
    const pot = new THREE.Mesh(new THREE.LatheGeometry(potPts, 36), potMat);
    pot.castShadow = true;
    pot.receiveShadow = true;
    plantGroup.add(pot);

    const soil = new THREE.Mesh(
      new THREE.CylinderGeometry(0.49, 0.47, 0.08, 28),
      new THREE.MeshStandardMaterial({ color: 0x3b2d22, roughness: 0.96 })
    );
    soil.position.y = 0.84;
    plantGroup.add(soil);

    const stemMat = new THREE.MeshStandardMaterial({ color: 0x3e5e38, roughness: 0.72 });
    const leafMat = new THREE.MeshPhysicalMaterial({
      color: 0x3f7a44,
      roughness: 0.36,
      metalness: 0.02,
      clearcoat: 0.32,
      clearcoatRoughness: 0.35,
      side: THREE.DoubleSide,
    });
    const leafGeo = createBotanicalLeafGeo(0.76, 0.38);

    const fronds = [
      { ang: 0.2, h: 1.55, spread: 0.65, scale: 1.08 },
      { ang: 1.05, h: 1.82, spread: 0.52, scale: 1.15 },
      { ang: 1.9, h: 1.48, spread: 0.72, scale: 0.96 },
      { ang: 2.75, h: 1.72, spread: 0.58, scale: 1.12 },
      { ang: 3.55, h: 1.4, spread: 0.68, scale: 0.92 },
      { ang: 4.35, h: 1.92, spread: 0.44, scale: 1.05 },
      { ang: 5.15, h: 1.5, spread: 0.7, scale: 1.0 },
      { ang: 5.85, h: 1.28, spread: 0.54, scale: 0.85 },
      { ang: 0.62, h: 2.05, spread: 0.26, scale: 0.94 },
    ];
    fronds.forEach((f) => {
      const tipX = Math.cos(f.ang) * f.spread;
      const tipZ = Math.sin(f.ang) * f.spread;
      const curve = new THREE.CatmullRomCurve3([
        new THREE.Vector3(tipX * 0.1, 0.84, tipZ * 0.1),
        new THREE.Vector3(tipX * 0.42, 0.84 + (f.h - 0.84) * 0.6, tipZ * 0.42),
        new THREE.Vector3(tipX, f.h, tipZ),
      ]);
      const stem = new THREE.Mesh(new THREE.TubeGeometry(curve, 14, 0.02, 8, false), stemMat);
      stem.castShadow = true;
      plantGroup.add(stem);

      const leaf = new THREE.Mesh(leafGeo, leafMat);
      leaf.position.set(tipX, f.h, tipZ);
      leaf.rotation.order = 'YXZ';
      leaf.rotation.y = -f.ang + Math.PI / 2;
      leaf.rotation.x = 0.88;
      leaf.scale.setScalar(f.scale);
      leaf.castShadow = true;
      leaf.receiveShadow = true;
      plantGroup.add(leaf);
    });

    // =========================================================================
    // 7. HOLLOWED STONEWARE ESPRESSO CUP & SCULPTED FOUNTAIN PEN
    // =========================================================================
    const mugGroup = new THREE.Group();
    mugGroup.position.set(2.72, 0.028, 0.92);
    scene.add(mugGroup);

    const coaster = new THREE.Mesh(
      new THREE.CylinderGeometry(0.45, 0.46, 0.028, 32),
      new THREE.MeshStandardMaterial({ color: 0xa47c55, roughness: 0.88 })
    );
    coaster.position.y = 0.014;
    coaster.receiveShadow = true;
    mugGroup.add(coaster);

    // True hollow stoneware mug profile via LatheGeometry
    const mugPts = [
      new THREE.Vector2(0.0, 0.028),
      new THREE.Vector2(0.3, 0.028),
      new THREE.Vector2(0.34, 0.68),
      new THREE.Vector2(0.32, 0.72),
      new THREE.Vector2(0.285, 0.71),
      new THREE.Vector2(0.265, 0.14),
      new THREE.Vector2(0.0, 0.14),
    ];
    const mugMat = new THREE.MeshStandardMaterial({ color: 0xf0ebe3, roughness: 0.36, metalness: 0.03 });
    const mugBody = new THREE.Mesh(new THREE.LatheGeometry(mugPts, 36), mugMat);
    mugBody.castShadow = true;
    mugBody.receiveShadow = true;
    mugGroup.add(mugBody);

    const coffeeTop = new THREE.Mesh(
      new THREE.CircleGeometry(0.28, 28),
      new THREE.MeshPhysicalMaterial({ color: 0x2d180c, roughness: 0.06, clearcoat: 1.0 })
    );
    coffeeTop.rotation.x = -Math.PI / 2;
    coffeeTop.position.y = 0.61;
    mugGroup.add(coffeeTop);

    const mugHandle = new THREE.Mesh(new THREE.TorusGeometry(0.18, 0.036, 14, 28, Math.PI), mugMat);
    mugHandle.position.set(0.31, 0.39, 0);
    mugHandle.rotation.z = -Math.PI / 2;
    mugHandle.castShadow = true;
    mugGroup.add(mugHandle);

    // Sculpted anodized pen with tapered nib + pocket clip
    const penGroup = new THREE.Group();
    penGroup.position.set(1.56, 0.064, 2.15);
    penGroup.rotation.y = -0.36;
    scene.add(penGroup);

    const penBodyMat = new THREE.MeshStandardMaterial({ color: 0x2a313d, roughness: 0.3, metalness: 0.75 });
    const penGoldMat = new THREE.MeshStandardMaterial({ color: 0xd4b270, roughness: 0.2, metalness: 0.92 });

    const penBarrel = new THREE.Mesh(new THREE.CylinderGeometry(0.034, 0.034, 1.1, 20), penBodyMat);
    penBarrel.rotation.z = Math.PI / 2;
    penBarrel.castShadow = true;
    penGroup.add(penBarrel);

    const penNib = new THREE.Mesh(new THREE.ConeGeometry(0.034, 0.22, 20), penGoldMat);
    penNib.rotation.z = Math.PI / 2;
    penNib.position.x = -0.66;
    penNib.castShadow = true;
    penGroup.add(penNib);

    const penBand = new THREE.Mesh(new THREE.CylinderGeometry(0.036, 0.036, 0.045, 20), penGoldMat);
    penBand.rotation.z = Math.PI / 2;
    penBand.position.x = 0.05;
    penGroup.add(penBand);

    const penClip = new THREE.Mesh(new THREE.BoxGeometry(0.36, 0.012, 0.022), penGoldMat);
    penClip.position.set(0.32, 0.04, 0);
    penGroup.add(penClip);

    // =========================================================================
    // 8. CLEAN ARCHITECTURAL GAZE MARKERS (Student + Guest/Jury)
    // =========================================================================
    this.studentMarker = this._createAvatarMarker(0x3b82f6, 'student');
    this.studentMarker.position.set(0.0, 3.65, 4.35);
    this.studentMarker.userData = { dragKind: 'student' };
    scene.add(this.studentMarker);
    this.draggables.push(this.studentMarker.children[0]);

    this.guestMarker = this._createAvatarMarker(0x10b981, 'guest');
    this.guestMarker.position.set(2.45, 3.55, 3.65);
    this.guestMarker.visible = false;
    this.guestMarker.userData = { dragKind: 'guest' };
    scene.add(this.guestMarker);
    this.draggables.push(this.guestMarker.children[0]);

    // =========================================================================
    // 9. SUNLIT ARCHITECTURAL LIGHTING RIG (DAY DEFAULT / GOLDEN / NIGHT)
    // =========================================================================
    this.hemiLight = new THREE.HemisphereLight(0xf6f9ff, 0xbfa58a, 1.15);
    scene.add(this.hemiLight);

    // Main directional sunlight streaming through the left tall window!
    this.keyLight = new THREE.DirectionalLight(0xfff9f0, 2.45);
    this.keyLight.position.set(-7.2, 7.6, 3.8);
    this.keyLight.castShadow = true;
    this.keyLight.shadow.mapSize.set(2048, 2048);
    this.keyLight.shadow.camera.near = 1;
    this.keyLight.shadow.camera.far = 24;
    this.keyLight.shadow.camera.left = -8.5;
    this.keyLight.shadow.camera.right = 8.5;
    this.keyLight.shadow.camera.top = 7.5;
    this.keyLight.shadow.camera.bottom = -7.5;
    this.keyLight.shadow.bias = -0.0004;
    this.keyLight.shadow.normalBias = 0.02;
    scene.add(this.keyLight);

    // Soft opposite room bounce fill light
    this.rimLight = new THREE.DirectionalLight(0xdbe8ff, 0.72);
    this.rimLight.position.set(6.5, 5.2, 4.5);
    scene.add(this.rimLight);

    this.setMood('day');
  }

  _createAvatarMarker(colorHex, dragKind) {
    const g = new THREE.Group();
    const orb = new THREE.Mesh(
      new THREE.SphereGeometry(0.19, 24, 24),
      new THREE.MeshPhysicalMaterial({
        color: colorHex,
        emissive: colorHex,
        emissiveIntensity: 0.25,
        roughness: 0.14,
        metalness: 0.08,
        clearcoat: 1.0,
        transparent: true,
        opacity: 0.62,
      })
    );
    orb.userData = { dragKind };
    g.add(orb);

    const halo = new THREE.Mesh(
      new THREE.TorusGeometry(0.26, 0.01, 12, 36),
      new THREE.MeshBasicMaterial({ color: colorHex, transparent: true, opacity: 0.45 })
    );
    halo.rotation.x = Math.PI / 2;
    g.add(halo);
    return g;
  }

  setMood(mood) {
    if (mood === 'day') {
      // Bright Sunlit Parisian / Apple ELEGNT Architectural Room!
      this.scene.background = new THREE.Color(0xe6e3dc);
      this.scene.fog = new THREE.FogExp2(0xe6e3dc, 0.016);
      this.wallMat.color.set(0xefece4);
      this.windowSkyMat.color.set(0xf5faff);
      this.sunShaftMat.color.set(0xfff7e6);
      this.sunShaftMat.opacity = 0.14;

      this.hemiLight.color.set(0xf8fbff);
      this.hemiLight.groundColor.set(0xbda282);
      this.hemiLight.intensity = 1.2;

      this.keyLight.color.set(0xfff8ee);
      this.keyLight.intensity = 2.45;
      this.keyLight.position.set(-7.2, 7.6, 3.8);

      this.rimLight.color.set(0xdde9ff);
      this.rimLight.intensity = 0.75;
    } else if (mood === 'golden') {
      // Warm Late-Afternoon Golden Hour Study
      this.scene.background = new THREE.Color(0x3a2d26);
      this.scene.fog = new THREE.FogExp2(0x3a2d26, 0.024);
      this.wallMat.color.set(0xcab49e);
      this.windowSkyMat.color.set(0xffbf7a);
      this.sunShaftMat.color.set(0xffb86b);
      this.sunShaftMat.opacity = 0.22;

      this.hemiLight.color.set(0xffd7ad);
      this.hemiLight.groundColor.set(0x523c2c);
      this.hemiLight.intensity = 0.78;

      this.keyLight.color.set(0xffb46b);
      this.keyLight.intensity = 2.15;
      this.keyLight.position.set(-8.2, 4.4, 3.6);

      this.rimLight.color.set(0x88aae0);
      this.rimLight.intensity = 0.65;
    } else {
      // Cozy Night Room — showcases the warm HiLight volumetric desk lamp dramatically!
      this.scene.background = new THREE.Color(0x0c1018);
      this.scene.fog = new THREE.FogExp2(0x0c1018, 0.032);
      this.wallMat.color.set(0x282f3c);
      this.windowSkyMat.color.set(0x162238);
      this.sunShaftMat.opacity = 0.0;

      this.hemiLight.color.set(0x6e8ec4);
      this.hemiLight.groundColor.set(0x1f1914);
      this.hemiLight.intensity = 0.36;

      this.keyLight.color.set(0xbfd4fa);
      this.keyLight.intensity = 0.5;
      this.keyLight.position.set(-6.5, 6.8, 3.8);

      this.rimLight.color.set(0x6e98df);
      this.rimLight.intensity = 0.68;
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
