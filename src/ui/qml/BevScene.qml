import QtQuick
import QtQuick.Shapes
import QtQuick3D
import QtQuick3D.AssetUtils

Item {
    id: root

    // Camera — 3rd-person chase, cluster-style (low, behind ego, looking down the road)
    property real pitchDeg: 13.0
    property real yawDeg: 0.0
    property real zoomFactor: 1.08
    property real camH: 5.0
    property real camDist: 10.5
    property real calibPitch: 0.0
    property real calibH: 1.0
    property real panX: 0.0
    property real panY: 0.0
    property bool cinematicRoad: true
    property bool showLaneLines: true
    property bool scenicView: false
    property bool showCalib: false
    // Metric debug overlay: ego centreline, 10 m ruler ticks, per-object (x, z)
    // labels in metres. Toggle from Python via _set("debugMetric", True).
    property bool debugMetric: false
    property real debugMaxZ: 70.0
    property real diskRadiusM: 70.0
    // Parsed trafficJson rows, kept for the metric labels above. Populated by
    // applyTraffic so the labels can never disagree with what was drawn.
    property var debugRows: []
    // Background: Auto (clock) | Day | Dusk | Night | Demo
    property string envMode: "auto"
    property real clockHour: 12.0
    property string envPhase: "day"
    property string envLabel: "Env Auto"
    property color skyTopColor: "#1a2740"
    property color skyMidColor: "#3a4a62"
    property color skyBotColor: "#6b7a90"
    property color clearCol: "#0e1218"
    property color lightCol: "#fff4e8"
    property color ambientCol: "#4a5564"
    property real lightBright: 1.35
    property real sunElev: 0.65
    property real sunAzim: 18.0
    property color sunColor: "#ffe2a8"
    property real sunGlowScale: 1.0
    property bool sunIsMoon: false
    property color asphaltCol: "#101216"
    property color shoulderCol: "#0a0c10"
    property color mistCol: "#6b7a90"
    property real mistOpacity: 0.55
    property color mountainFarCol: "#3a4a5c"
    property color mountainNearCol: "#1e2834"
    property real sunPosX: 20
    property real sunPosY: 40
    property real sunPosZ: -90
    property string cipoStatus: "SAFE"
    property bool cipoVisible: false
    property real cipoX: 0
    property real cipoZ: -12
    property real cipoDist: 0
    readonly property color cipoGlow: cipoStatus === "DANGER" ? "#e23a3c"
                                     : (cipoStatus === "WARNING" ? "#e09a20" : "#3ec8ff")
    readonly property color corridorColor: cipoStatus === "DANGER" ? Qt.rgba(0.95, 0.18, 0.32, 0.80)
                                     : (cipoStatus === "WARNING" ? Qt.rgba(1.0, 0.72, 0.08, 0.78)
                                                                 : Qt.rgba(0.76, 0.93, 0.14, 0.82))
    property string overlayHint: "Phase 2 — loading GLB"
    property url egoGltf: ""
    property url skodaGltf: ""
    property url teslaGltf: ""
    property url shcGltf: ""
    property url dodgeGltf: ""
    property real egoScale: 1.0
    property real skodaScale: 100.0
    property real teslaScale: 1.0
    property real teslaRotX: -90
    property real teslaRotY: 0
    property real teslaRotZ: 0
    property real teslaY: 0
    property bool teslaFitted: false
    property real shcScale: 0.01
    property real shcRotY: 90
    property real dodgeScale: 0.05
    property real egoRotX: 0
    property real egoRotY: 180
    property real egoRotZ: 0
    property real egoY: 0.3
    // Lane-anchored render: the lane is pinned and the ego moves within it.
    property real egoX: 0
    property real egoYawDeg: 0
    property real worldYawDeg: 0
    property bool laneValid: false
    property bool laneHeld: false
    property real targetCarLength: 4.6
    property real targetTruckLength: 10.0
    property bool egoFitted: false
    property bool skodaFitted: false
    property bool shcFitted: false
    property bool dodgeFitted: false
    property real skodaY: 0
    property real shcY: 0
    property real dodgeY: 0
    property string egoDebug: "bounds pending"
    property string trafficJson: "[]"
    property int trafficCount: 0
    property string corridorJson: "[]"
    property string laneJson: "[]"
    property string dashJson: "[]"
    property string edgeJson: "[]"
    property string trailJson: "[]"

    onTrafficJsonChanged: applyTraffic()
    onTeslaYChanged: applyTraffic()
    onSkodaYChanged: applyTraffic()
    onShcYChanged: applyTraffic()
    onDodgeYChanged: applyTraffic()
    onCorridorJsonChanged: applySegPool(corrRep, root.corridorJson, 0.045)
    onLaneJsonChanged: applyLanes()
    onDashJsonChanged: applyDashes()
    onEdgeJsonChanged: applyEdges()
    onTrailJsonChanged: applyTrail()
    onShowLaneLinesChanged: { applyLanes(); applyDashes(); applyEdges(); applyTrail() }
    onCinematicRoadChanged: { applyLanes(); applyDashes(); applyEdges(); applyTrail() }
    onEnvModeChanged: refreshEnv()
    onClockHourChanged: refreshEnv()

    Component.onCompleted: {
        syncClockHour()
        refreshEnv()
        applySegPool(corrRep, root.corridorJson, 0.045)
        applyLanes()
        applyDashes()
        applyEdges()
        applyTrail()
    }

    function syncClockHour() {
        const d = new Date()
        root.clockHour = d.getHours() + d.getMinutes() / 60.0 + d.getSeconds() / 3600.0
    }

    function cycleEnvMode() {
        const modes = ["auto", "day", "dusk", "night", "demo"]
        const i = modes.indexOf(root.envMode)
        root.envMode = modes[(i < 0 ? 0 : i + 1) % modes.length]
    }

    function _lerp(a, b, t) { return a + (b - a) * t }

    function _lerpColor(a, b, t) {
        return Qt.rgba(
            a.r + (b.r - a.r) * t,
            a.g + (b.g - a.g) * t,
            a.b + (b.b - a.b) * t,
            1.0
        )
    }

    function _palette(phase, demoBoost) {
        // Soft sky + mist + sun look; demoBoost amplifies saturation/brightness.
        if (phase === "night") {
            return {
                top: "#050810", mid: "#0b1220", bot: "#1a2740",
                clear: "#050810",
                light: "#c8d6e8", ambient: "#1c2430",
                bright: demoBoost ? 0.55 : 0.42,
                sun: "#d8e6ff", asphalt: "#0a0c10", shoulder: "#07090c",
                mist: "#1a2740", mistOp: 0.42, moon: true, glow: demoBoost ? 1.35 : 1.0,
                mtnFar: "#121820", mtnNear: "#080c10"
            }
        }
        if (phase === "dusk") {
            return {
                top: "#1a1038", mid: "#b04838", bot: "#f0a060",
                clear: "#1a1028",
                light: "#ffd0a8", ambient: "#5a4050",
                bright: demoBoost ? 1.55 : 1.15,
                sun: "#ffb060", asphalt: "#121018", shoulder: "#0e0c12",
                mist: "#e09060", mistOp: demoBoost ? 0.55 : 0.40, moon: false,
                glow: demoBoost ? 1.8 : 1.25,
                mtnFar: "#2a1830", mtnNear: "#120c18"
            }
        }
        // day — stronger top→horizon contrast so the sky does not read as flat fill
        return {
            top: "#1e5aaa", mid: "#5a9ed8", bot: "#c5e2f8",
            clear: "#5a90c0",
            light: "#fff8ee", ambient: "#6a7a88",
            bright: demoBoost ? 1.75 : 1.40,
            sun: "#ffe8a0", asphalt: "#12161c", shoulder: "#101812",
            mist: "#b8d4ea", mistOp: demoBoost ? 0.45 : 0.32, moon: false,
            glow: demoBoost ? 1.55 : 1.1,
            mtnFar: "#3d5568", mtnNear: "#1c2832"
        }
    }

    function _phaseFromHour(h) {
        // Night 19–6, dusk 15–19, day otherwise (morning shares day palette).
        if (h >= 19.0 || h < 6.0)
            return "night"
        if (h >= 15.0)
            return "dusk"
        return "day"
    }

    function _asCol(c) {
        return (typeof c === "object") ? c : Qt.color(c)
    }

    function _blendPalettes(a, b, t) {
        return {
            top: _lerpColor(_asCol(a.top), _asCol(b.top), t),
            mid: _lerpColor(_asCol(a.mid), _asCol(b.mid), t),
            bot: _lerpColor(_asCol(a.bot), _asCol(b.bot), t),
            clear: _lerpColor(_asCol(a.clear), _asCol(b.clear), t),
            light: _lerpColor(_asCol(a.light), _asCol(b.light), t),
            ambient: _lerpColor(_asCol(a.ambient), _asCol(b.ambient), t),
            bright: _lerp(a.bright, b.bright, t),
            sun: _lerpColor(_asCol(a.sun), _asCol(b.sun), t),
            asphalt: _lerpColor(_asCol(a.asphalt), _asCol(b.asphalt), t),
            shoulder: _lerpColor(_asCol(a.shoulder), _asCol(b.shoulder), t),
            mist: _lerpColor(_asCol(a.mist), _asCol(b.mist), t),
            mistOp: _lerp(a.mistOp, b.mistOp, t),
            moon: t > 0.5 ? b.moon : a.moon,
            glow: _lerp(a.glow, b.glow, t),
            mtnFar: _lerpColor(_asCol(a.mtnFar), _asCol(b.mtnFar), t),
            mtnNear: _lerpColor(_asCol(a.mtnNear), _asCol(b.mtnNear), t)
        }
    }

    function _autoPalette(h) {
        // Smooth crossfades near phase boundaries so Auto does not hard-cut.
        if (h >= 5.0 && h < 7.0)
            return _blendPalettes(_palette("night", false), _palette("day", false), (h - 5.0) / 2.0)
        if (h >= 14.5 && h < 16.5)
            return _blendPalettes(_palette("day", false), _palette("dusk", false), (h - 14.5) / 2.0)
        if (h >= 18.0 && h < 20.0)
            return _blendPalettes(_palette("dusk", false), _palette("night", false), (h - 18.0) / 2.0)
        return _palette(_phaseFromHour(h), false)
    }

    function _applySunOrbit(h, pal) {
        // Day arc 06–18; night moon arc 18–06 (wrapped).
        let elev = 0.35
        let azim = 0.0
        if (!pal.moon) {
            const p = Math.max(0.0, Math.min(1.0, (h - 6.0) / 12.0))
            elev = Math.sin(Math.PI * p)
            azim = _lerp(-58.0, 58.0, p)
        } else {
            let hn = h
            if (hn < 6.0)
                hn += 24.0
            const p = Math.max(0.0, Math.min(1.0, (hn - 18.0) / 12.0))
            elev = 0.28 + 0.35 * Math.sin(Math.PI * p)
            azim = _lerp(50.0, -50.0, p)
        }
        root.sunElev = elev
        root.sunAzim = azim
        root.sunIsMoon = !!pal.moon
        root.sunGlowScale = pal.glow
        root.sunColor = (typeof pal.sun === "object") ? pal.sun : Qt.color(pal.sun)

        const dist = 98.0
        const elevDeg = 6.0 + elev * (pal.moon ? 32.0 : 48.0)
        const az = azim * Math.PI / 180.0
        const el = elevDeg * Math.PI / 180.0
        root.sunPosX = Math.sin(az) * dist * Math.cos(el)
        root.sunPosY = Math.sin(el) * dist
        root.sunPosZ = -Math.cos(az) * dist * Math.cos(el)
    }

    function refreshEnv() {
        const mode = root.envMode
        const demo = (mode === "demo")
        let phase = "day"
        let pal
        if (mode === "auto") {
            phase = _phaseFromHour(root.clockHour)
            pal = _autoPalette(root.clockHour)
            root.envLabel = "Env Auto"
        } else if (mode === "demo") {
            // Demo locks a rich golden-hour look for recordings.
            phase = "dusk"
            pal = _palette("dusk", true)
            root.envLabel = "Env Demo"
        } else {
            phase = mode
            pal = _palette(mode, false)
            root.envLabel = "Env " + mode.charAt(0).toUpperCase() + mode.slice(1)
        }
        root.envPhase = phase

        const asColor = _asCol
        root.skyTopColor = asColor(pal.top)
        root.skyMidColor = asColor(pal.mid)
        root.skyBotColor = asColor(pal.bot)
        root.clearCol = asColor(pal.clear)
        root.lightCol = asColor(pal.light)
        root.ambientCol = asColor(pal.ambient)
        root.lightBright = pal.bright
        root.asphaltCol = asColor(pal.asphalt)
        root.shoulderCol = asColor(pal.shoulder)
        root.mistCol = asColor(pal.mist)
        root.mistOpacity = pal.mistOp
        root.mountainFarCol = asColor(pal.mtnFar)
        root.mountainNearCol = asColor(pal.mtnNear)

        // Sun hour: Auto uses clock; fixed presets use a canonical hour.
        let h = root.clockHour
        if (mode === "day")
            h = 12.0
        else if (mode === "dusk" || mode === "demo")
            h = 17.2
        else if (mode === "night")
            h = 22.0
        _applySunOrbit(h, {
            moon: pal.moon,
            glow: pal.glow,
            sun: pal.sun
        })
    }

    Timer {
        interval: 30000
        running: true
        repeat: true
        onTriggered: {
            if (root.envMode === "auto")
                root.syncClockHour()
        }
    }

    // Screen-space sun/moon (azimuth → X, elevation → height above horizon band).
    readonly property real sunDiscSize: (root.sunIsMoon ? 22 : 28) * root.sunGlowScale
    readonly property real sunScreenX: width * (0.5 + root.sunAzim / 145.0) - sunDiscSize * 1.3
    // Keep the disc in the visible sky band (above the road horizon ~40% down).
    readonly property real sunScreenY: {
        const y = height * (0.33 - root.sunElev * 0.20) - sunDiscSize * 1.3
        return Math.max(6, Math.min(height * 0.36, y))
    }

    // Cluster void — no sky/sun/buildings. Matches the OEM dashboards.
    Rectangle {
        id: skyBackdrop
        anchors.fill: parent
        z: 0
        gradient: Gradient {
            GradientStop { position: 0.0; color: "#070b12" }
            GradientStop { position: 0.42; color: "#0c141e" }
            GradientStop { position: 1.0; color: "#101820" }
        }
    }

    Item {
        id: sunGlow
        z: 1
        visible: false
        width: 1
        height: 1
    }

    // Distant mountain ranges along the horizon (replaces the fake mist slab).
    Item {
        id: mountainLayer
        anchors.left: parent.left
        anchors.right: parent.right
        z: 1
        y: parent.height * 0.26
        height: parent.height * 0.22
        opacity: 0.0
        visible: false

        // Far range — lighter, softer, sits behind near peaks
        Shape {
            anchors.fill: parent
            ShapePath {
                fillColor: root.mountainFarCol
                strokeWidth: 0
                startX: 0
                startY: mountainLayer.height
                PathLine { x: mountainLayer.width * 0.00; y: mountainLayer.height * 0.72 }
                PathLine { x: mountainLayer.width * 0.10; y: mountainLayer.height * 0.38 }
                PathLine { x: mountainLayer.width * 0.18; y: mountainLayer.height * 0.55 }
                PathLine { x: mountainLayer.width * 0.28; y: mountainLayer.height * 0.22 }
                PathLine { x: mountainLayer.width * 0.38; y: mountainLayer.height * 0.48 }
                PathLine { x: mountainLayer.width * 0.50; y: mountainLayer.height * 0.18 }
                PathLine { x: mountainLayer.width * 0.62; y: mountainLayer.height * 0.42 }
                PathLine { x: mountainLayer.width * 0.74; y: mountainLayer.height * 0.12 }
                PathLine { x: mountainLayer.width * 0.86; y: mountainLayer.height * 0.40 }
                PathLine { x: mountainLayer.width * 0.96; y: mountainLayer.height * 0.28 }
                PathLine { x: mountainLayer.width * 1.00; y: mountainLayer.height * 0.58 }
                PathLine { x: mountainLayer.width; y: mountainLayer.height }
                PathLine { x: 0; y: mountainLayer.height }
            }
        }

        // Near range — darker silhouette in front
        Shape {
            anchors.fill: parent
            ShapePath {
                fillColor: root.mountainNearCol
                strokeWidth: 0
                startX: 0
                startY: mountainLayer.height
                PathLine { x: mountainLayer.width * 0.00; y: mountainLayer.height * 0.88 }
                PathLine { x: mountainLayer.width * 0.08; y: mountainLayer.height * 0.58 }
                PathLine { x: mountainLayer.width * 0.16; y: mountainLayer.height * 0.70 }
                PathLine { x: mountainLayer.width * 0.26; y: mountainLayer.height * 0.45 }
                PathLine { x: mountainLayer.width * 0.36; y: mountainLayer.height * 0.68 }
                PathLine { x: mountainLayer.width * 0.48; y: mountainLayer.height * 0.52 }
                PathLine { x: mountainLayer.width * 0.58; y: mountainLayer.height * 0.72 }
                PathLine { x: mountainLayer.width * 0.70; y: mountainLayer.height * 0.40 }
                PathLine { x: mountainLayer.width * 0.82; y: mountainLayer.height * 0.62 }
                PathLine { x: mountainLayer.width * 0.92; y: mountainLayer.height * 0.50 }
                PathLine { x: mountainLayer.width * 1.00; y: mountainLayer.height * 0.78 }
                PathLine { x: mountainLayer.width; y: mountainLayer.height }
                PathLine { x: 0; y: mountainLayer.height }
            }
        }

        // Soft atmospheric wash over the foothills (not a hard slab)
        Rectangle {
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.bottom: parent.bottom
            height: parent.height * 0.55
            gradient: Gradient {
                GradientStop {
                    position: 0.0
                    color: Qt.rgba(root.mistCol.r, root.mistCol.g, root.mistCol.b, 0.0)
                }
                GradientStop {
                    position: 0.55
                    color: Qt.rgba(root.mistCol.r, root.mistCol.g, root.mistCol.b,
                                   root.mistOpacity * 0.35)
                }
                GradientStop {
                    position: 1.0
                    color: Qt.rgba(root.mistCol.r, root.mistCol.g, root.mistCol.b,
                                   root.mistOpacity * 0.55)
                }
            }
        }
    }

    function applyLanes() {
        if (!root.showLaneLines) {
            applySegPool(laneRep, "[]", 0.05)
            return
        }
        if (!root.cinematicRoad) {
            applySegPool(laneRep, root.laneJson, 0.05)
            return
        }
        // Film: ego-adjacent (pal 0) is drawn as white dashes instead.
        let rows = []
        try { rows = JSON.parse(root.laneJson || "[]") } catch (e) { rows = [] }
        rows = rows.filter(function (r) { return (r.c | 0) !== 0 })
        applySegPool(laneRep, JSON.stringify(rows), 0.05)
    }

    function applyDashes() {
        const src = (root.cinematicRoad && root.dashJson) ? root.dashJson : "[]"
        applySegPool(dashRep, src, 0.03)
    }

    function applyEdges() {
        // Road edges come only from stable, hysteretic lane slots. There is no
        // static ±5.35 fallback: it used to pop in on every detection miss and
        // cross the real edge on curves.
        applySegPool(edgeRep, root.cinematicRoad ? root.edgeJson : "[]", 0.028)
    }

    function applyTrail() {
        applySegPool(trailRep, "[]", 0.02)
    }

    function applyTraffic() {
        let rows = []
        try { rows = JSON.parse(root.trafficJson || "[]") } catch (e) { rows = [] }
        root.debugRows = rows
        const pools = [
            [skoda0, skoda1, skoda2, skoda3, skoda4, skoda5, skoda6, skoda7],
            [tesla0],
            [shc0],
            [truck0, truck1]
        ]
        const occupied = [
            [false, false, false, false, false, false, false, false],
            [false],
            [false],
            [false, false]
        ]
        for (let i = 0; i < rows.length; i++) {
            const r = rows[i]
            let kind = r.kind | 0
            let slot = r.slot | 0
            if (kind < 0 || kind >= pools.length)
                continue
            if (slot < 0 || slot >= pools[kind].length)
                continue
            const node = pools[kind][slot]
            node.visible = true
            occupied[kind][slot] = true
            const py = (kind === 1) ? root.teslaY
                     : (kind === 0) ? root.skodaY
                     : (kind === 2) ? root.shcY
                     : root.dodgeY
            node.x = r.posX
            node.y = py
            node.z = r.posZ
            node.eulerRotation = Qt.vector3d(0, (r.yawDeg !== undefined ? r.yawDeg : root.egoRotY), 0)
            // Far cars otherwise shrink to specks in the chase camera.
            const z = Math.max(0.0, -r.posZ)
            const boost = (z > 18.0) ? Math.min(1.9, 1.0 + 0.016 * (z - 18.0)) : 1.0
            node.scale = Qt.vector3d(boost, boost, boost)
        }
        for (let k = 0; k < pools.length; k++) {
            for (let j = 0; j < pools[k].length; j++) {
                if (!occupied[k][j]) {
                    pools[k][j].visible = false
                    pools[k][j].scale = Qt.vector3d(1, 1, 1)
                }
            }
        }
        root.trafficCount = rows.length
    }

    function applySegPool(rep, jsonStr, yLift) {
        let rows = []
        try { rows = JSON.parse(jsonStr || "[]") } catch (e) { rows = [] }
        const n = rep.count
        for (let i = 0; i < n; i++) {
            const m = rep.objectAt(i)
            if (!m)
                continue
            if (i < rows.length) {
                const r = rows[i]
                m.visible = true
                m.position = Qt.vector3d(r.x, yLift, r.z)
                m.eulerRotation = Qt.vector3d(0, r.yaw, 0)
                m.scale = Qt.vector3d((r.w || 0.18) / 100.0, 0.00022, (r.len || 2.5) / 100.0)
                if (m.pal !== undefined)
                    m.pal = r.c | 0
            } else {
                m.visible = false
            }
        }
    }

    readonly property real lookPitch: -(pitchDeg + calibPitch)
    readonly property real camHeight: camH * (calibH / 1.6)
    readonly property real camZ: camDist / Math.max(zoomFactor, 0.15)
    readonly property real yawRad: yawDeg * Math.PI / 180.0

    function fitEgoFromBounds() {
        const mn = egoCar.bounds.minimum
        const mx = egoCar.bounds.maximum
        const dx = Math.abs(mx.x - mn.x)
        const dy = Math.abs(mx.y - mn.y)
        const dz = Math.abs(mx.z - mn.z)
        const longest = Math.max(dx, dy, dz)
        root.egoDebug = "raw " + dx.toFixed(3) + "×" + dy.toFixed(3) + "×" + dz.toFixed(3)
        if (longest < 1e-4)
            return
        if (longest < 1.5) {
            const s = root.targetCarLength / longest
            root.egoScale = s
            root.egoY = -mn.y * s
            root.egoDebug += "  → scale " + s.toFixed(1)
            // Never copy Audi fit onto traffic — Skoda/SHC/Dodge have their own units.
        } else if (longest > 20.0) {
            const s = root.targetCarLength / longest
            root.egoScale = s
            root.egoY = -mn.y * s
            root.egoDebug += "  → shrink " + s.toFixed(3)
        } else {
            root.egoDebug += "  keep scale " + root.egoScale.toFixed(1)
        }
        root.egoFitted = true
        console.log("[BEV] ego fit", root.egoDebug)
    }

    function fitMeshToCarLength(loader, targetLen) {
        if (!loader || loader.status !== RuntimeLoader.Success)
            return null
        const mn = loader.bounds.minimum
        const mx = loader.bounds.maximum
        const dx = Math.abs(mx.x - mn.x)
        const dy = Math.abs(mx.y - mn.y)
        const dz = Math.abs(mx.z - mn.z)
        const longest = Math.max(dx, dy, dz)
        if (longest < 1e-4)
            return null
        const want = (targetLen === undefined || targetLen === null) ? root.targetCarLength : targetLen
        const s = want / longest
        return { s: s, y: -Math.min(mn.y, mx.y) * s, longest: longest }
    }

    function fitSkodaFromBounds() {
        const f = fitMeshToCarLength(skodaLoader)
        if (!f)
            return
        root.skodaScale = f.s
        root.skodaY = f.y
        root.skodaFitted = true
        console.log("[BEV] skoda fit longest", f.longest.toFixed(3), "scale", f.s.toFixed(3), "y", f.y.toFixed(3))
    }

    function fitShcFromBounds() {
        const f = fitMeshToCarLength(shcLoader)
        if (!f)
            return
        root.shcScale = f.s
        root.shcY = f.y
        root.shcFitted = true
        console.log("[BEV] shc fit longest", f.longest.toFixed(3), "scale", f.s.toFixed(3), "y", f.y.toFixed(3))
    }

    function fitDodgeFromBounds() {
        const f = fitMeshToCarLength(dodgeLoader, root.targetTruckLength)
        if (!f)
            return
        root.dodgeScale = f.s
        root.dodgeY = f.y
        root.dodgeFitted = true
        console.log("[BEV] truck fit longest", f.longest.toFixed(3), "scale", f.s.toFixed(3), "y", f.y.toFixed(3))
    }

    function fitTeslaFromBounds() {
        const mn = teslaCar.bounds.minimum
        const mx = teslaCar.bounds.maximum
        const dx = Math.abs(mx.x - mn.x)
        const dy = Math.abs(mx.y - mn.y)
        const dz = Math.abs(mx.z - mn.z)
        const longest = Math.max(dx, dy, dz)
        if (longest < 1e-4)
            return
        const s = root.targetCarLength / longest
        root.teslaScale = s
        // teslaFix is Rx(-90): local (x,y,z) → (x, z, -y). World up is local Z.
        const worldYmin = Math.min(mn.z, mx.z)
        root.teslaY = -worldYmin * s
        root.teslaFitted = true
        console.log("[BEV] tesla fit raw", dx.toFixed(3), dy.toFixed(3), dz.toFixed(3),
                    "mn", mn.x.toFixed(2), mn.y.toFixed(2), mn.z.toFixed(2),
                    "→ scale", s.toFixed(3), "y", root.teslaY.toFixed(3))
    }

    View3D {
        id: view
        anchors.fill: parent
        z: 2
        camera: sceneCamera

        environment: SceneEnvironment {
            // Let the 2D sky/sun behind this view show through empty pixels.
            backgroundMode: SceneEnvironment.Transparent
            clearColor: "#00000000"
            antialiasingMode: SceneEnvironment.NoAA
        }

        PerspectiveCamera {
            id: sceneCamera
            // Orbit around ego: move the camera, don't just yaw in place.
            position: Qt.vector3d(
                root.panX + Math.sin(root.yawRad) * root.camZ,
                root.camHeight + root.panY,
                Math.cos(root.yawRad) * root.camZ
            )
            eulerRotation: Qt.vector3d(root.lookPitch, root.yawDeg, 0)
            fieldOfView: 44
            clipNear: 0.3
            clipFar: 180
        }

        DirectionalLight {
            eulerRotation.x: -32
            eulerRotation.y: 22
            brightness: 0.92
            color: "#d4e2f0"
            castsShadow: false
            ambientColor: "#243040"
        }

        // 3rd-person cluster road: long vanishing-point strip, ego-lane only.
        Node {
            id: worldRig
            eulerRotation: Qt.vector3d(0, root.worldYawDeg, 0)

            // Dark void under the road so the sides fall off into the cluster.
            Model {
                source: "#Rectangle"
                eulerRotation.x: -90
                scale: Qt.vector3d(0.80, 1.20, 1)
                position: Qt.vector3d(0, -0.04, -40)
                materials: PrincipledMaterial {
                    lighting: PrincipledMaterial.NoLighting
                    baseColor: "#0a1018"
                    roughness: 1.0
                }
            }
            // Pavement going to the horizon (about 12 m wide × 90 m long).
            Model {
                source: "#Rectangle"
                eulerRotation.x: -90
                scale: Qt.vector3d(0.12, 0.90, 1)
                position: Qt.vector3d(0, 0, -40)
                materials: PrincipledMaterial {
                    baseColor: "#1a2433"
                    roughness: 0.97
                    metalness: 0.0
                }
            }
            // BMW-style perspective grid on the pavement.
            Repeater3D {
                model: 16
                Model {
                    source: "#Cube"
                    position: Qt.vector3d(0, 0.02, 6.0 - index * 6.0)
                    scale: Qt.vector3d(0.12, 0.00008, 0.00018)
                    materials: PrincipledMaterial {
                        lighting: PrincipledMaterial.NoLighting
                        baseColor: "#2a4a62"
                        opacity: 0.28
                        alphaMode: PrincipledMaterial.Blend
                    }
                }
            }
            Repeater3D {
                model: 7
                Model {
                    source: "#Cube"
                    position: Qt.vector3d((index - 3) * 2.0, 0.02, -38)
                    scale: Qt.vector3d(0.00016, 0.00008, 0.90)
                    materials: PrincipledMaterial {
                        lighting: PrincipledMaterial.NoLighting
                        baseColor: "#2a4a62"
                        opacity: 0.22
                        alphaMode: PrincipledMaterial.Blend
                    }
                }
            }

        Repeater3D {
            id: edgeRep
            model: 24
            Model {
                visible: false
                source: "#Cube"
                materials: PrincipledMaterial {
                    lighting: PrincipledMaterial.NoLighting
                    baseColor: "#ebefff"
                }
            }
        }

        Repeater3D {
            id: dashRep
            model: 28
            Model {
                visible: false
                source: "#Cube"
                materials: PrincipledMaterial {
                    lighting: PrincipledMaterial.NoLighting
                    baseColor: "#c5d0dc"
                }
            }
        }

        Repeater3D {
            id: trailRep
            model: 32
            Model {
                property int pal: 0
                visible: false
                source: "#Cube"
                materials: PrincipledMaterial {
                    lighting: PrincipledMaterial.NoLighting
                    baseColor: pal === 1 ? "#8aa4c8" : "#6e889e"
                    opacity: 0.55
                    alphaMode: PrincipledMaterial.Blend
                }
            }
        }

        // ---- metric debug overlay -------------------------------------
        // Ego centreline at x=0 plus 10 m ruler ticks, in scene metres. This
        // is the ground truth against which trafficJson positions are read:
        // if a car sits between the 30 and 40 m ticks its posZ really is -3x.
        Model {
            source: "#Cube"
            visible: root.debugMetric
            position: Qt.vector3d(0, 0.06, -0.5 * root.debugMaxZ)
            scale: Qt.vector3d(0.0004, 0.0002, root.debugMaxZ / 100.0)
            materials: PrincipledMaterial {
                lighting: PrincipledMaterial.NoLighting
                baseColor: "#ff3b30"
                opacity: 0.75
                alphaMode: PrincipledMaterial.Blend
            }
        }
        Repeater3D {
            model: Math.floor(root.debugMaxZ / 10) + 1
            Model {
                source: "#Cube"
                visible: root.debugMetric && index > 0
                position: Qt.vector3d(0, 0.06, -index * 10.0)
                // 8 m wide tick, 12 cm deep: reads as a rung on the centreline.
                scale: Qt.vector3d(0.08, 0.0002, 0.0012)
                materials: PrincipledMaterial {
                    lighting: PrincipledMaterial.NoLighting
                    baseColor: "#ff9500"
                    opacity: 0.65
                    alphaMode: PrincipledMaterial.Blend
                }
            }
        }

        Repeater3D {
            id: corrRep
            model: 18
            Model {
                visible: false
                source: "#Cube"
                materials: PrincipledMaterial {
                    lighting: PrincipledMaterial.NoLighting
                    baseColor: root.corridorColor
                    opacity: 0.78
                    alphaMode: PrincipledMaterial.Blend
                }
            }
        }
        Repeater3D {
            id: laneRep
            model: 36
            Model {
                property int pal: 0
                visible: false
                source: "#Cube"
                materials: PrincipledMaterial {
                    lighting: PrincipledMaterial.NoLighting
                    baseColor: pal === 1 ? "#b89440" : (pal === 2 ? "#7aa8b4" : "#78dcff")
                }
            }
        }

        Model {
            id: cipoRing
            visible: false
            source: "#Cylinder"
            position: Qt.vector3d(root.cipoX, 0.06, root.cipoZ)
            scale: Qt.vector3d(0.034, 0.0004, 0.034)
            materials: PrincipledMaterial {
                lighting: PrincipledMaterial.NoLighting
                baseColor: root.cipoGlow
                opacity: 0.85
                alphaMode: PrincipledMaterial.Blend
            }
        }
        Model {
            id: cipoBeacon
            visible: false
            source: "#Cylinder"
            position: Qt.vector3d(root.cipoX, 1.35, root.cipoZ)
            scale: Qt.vector3d(0.0014, 0.026, 0.0014)
            materials: PrincipledMaterial {
                lighting: PrincipledMaterial.NoLighting
                baseColor: root.cipoGlow
            }
        }
        Node {
            visible: false
            eulerRotation: Qt.vector3d(root.lookPitch, root.yawDeg, 0)
            Model {
                source: "#Rectangle"
                scale: Qt.vector3d(0.020, 0.0055, 1)
                materials: PrincipledMaterial {
                    lighting: PrincipledMaterial.NoLighting
                    baseColor: root.cipoGlow
                }
            }
        }

        // Pooled traffic: one RuntimeLoader per slot, source set once.
        component TrafficRig: Node {
            visible: false
            Behavior on x { NumberAnimation { duration: 80; easing.type: Easing.Linear } }
            Behavior on z { NumberAnimation { duration: 80; easing.type: Easing.Linear } }
        }
        TrafficRig {
            id: skoda0
            visible: false
            RuntimeLoader {
                id: skodaLoader
                source: root.skodaGltf
                scale: Qt.vector3d(root.skodaScale, root.skodaScale, root.skodaScale)
            }
        }
        TrafficRig {
            id: skoda1
            visible: false
            RuntimeLoader {
                source: root.skodaGltf
                scale: Qt.vector3d(root.skodaScale, root.skodaScale, root.skodaScale)
            }
        }
        TrafficRig {
            id: skoda2
            visible: false
            RuntimeLoader {
                source: root.skodaGltf
                scale: Qt.vector3d(root.skodaScale, root.skodaScale, root.skodaScale)
            }
        }
        TrafficRig {
            id: skoda3
            visible: false
            RuntimeLoader {
                source: root.skodaGltf
                scale: Qt.vector3d(root.skodaScale, root.skodaScale, root.skodaScale)
            }
        }
        TrafficRig {
            id: skoda4
            visible: false
            RuntimeLoader {
                source: root.skodaGltf
                scale: Qt.vector3d(root.skodaScale, root.skodaScale, root.skodaScale)
            }
        }
        TrafficRig {
            id: skoda5
            visible: false
            RuntimeLoader {
                source: root.skodaGltf
                scale: Qt.vector3d(root.skodaScale, root.skodaScale, root.skodaScale)
            }
        }
        TrafficRig {
            id: skoda6
            visible: false
            RuntimeLoader {
                source: root.skodaGltf
                scale: Qt.vector3d(root.skodaScale, root.skodaScale, root.skodaScale)
            }
        }
        TrafficRig {
            id: skoda7
            visible: false
            RuntimeLoader {
                source: root.skodaGltf
                scale: Qt.vector3d(root.skodaScale, root.skodaScale, root.skodaScale)
            }
        }
        TrafficRig {
            id: tesla0
            visible: false
            // Inner node: cancel Tesla GLB extra X-90. Outer node yaw = ego (180).
            Node {
                id: teslaFix
                eulerRotation: Qt.vector3d(root.teslaRotX, root.teslaRotY, root.teslaRotZ)
                RuntimeLoader {
                    id: teslaCar
                    source: root.teslaGltf
                    scale: Qt.vector3d(root.teslaScale, root.teslaScale, root.teslaScale)
                }
            }
        }

        Timer {
            id: teslaFitTimer
            interval: 33
            repeat: true
            running: teslaCar.status === RuntimeLoader.Success && !root.teslaFitted
            onTriggered: root.fitTeslaFromBounds()
        }
        Timer {
            id: skodaFitTimer
            interval: 33
            repeat: true
            running: skodaLoader.status === RuntimeLoader.Success && !root.skodaFitted
            onTriggered: root.fitSkodaFromBounds()
        }
        TrafficRig {
            id: shc0
            visible: false
            RuntimeLoader {
                id: shcLoader
                source: root.shcGltf
                scale: Qt.vector3d(root.shcScale, root.shcScale, root.shcScale)
                eulerRotation: Qt.vector3d(0, root.shcRotY, 0)
            }
        }
        Timer {
            id: shcFitTimer
            interval: 33
            repeat: true
            running: shcLoader.status === RuntimeLoader.Success && !root.shcFitted
            onTriggered: root.fitShcFromBounds()
        }
        TrafficRig {
            id: truck0
            visible: false
            RuntimeLoader {
                id: dodgeLoader
                source: root.dodgeGltf
                scale: Qt.vector3d(root.dodgeScale, root.dodgeScale, root.dodgeScale)
            }
        }
        TrafficRig {
            id: truck1
            visible: false
            RuntimeLoader {
                source: root.dodgeGltf
                scale: Qt.vector3d(root.dodgeScale, root.dodgeScale, root.dodgeScale)
            }
        }
        Timer {
            id: dodgeFitTimer
            interval: 33
            repeat: true
            running: dodgeLoader.status === RuntimeLoader.Success && !root.dodgeFitted
            onTriggered: root.fitDodgeFromBounds()
        }
        } // worldRig — live world flashes opposite the ego, then settles

        // Ego stays heading-up except for the brief turn yaw on this GLB.
        Model {
            visible: root.egoGltf.toString() === "" || egoCar.status === RuntimeLoader.Error
            source: "#Cube"
            position: Qt.vector3d(root.egoX, 0.65, 0)
            eulerRotation: Qt.vector3d(0, root.egoYawDeg, 0)
            scale: Qt.vector3d(0.018, 0.013, 0.043)
            materials: PrincipledMaterial {
                baseColor: "#00c8ff"
                roughness: 0.35
                metalness: 0.15
            }
        }
        RuntimeLoader {
            id: egoCar
            source: root.egoGltf
            visible: root.egoGltf.toString() !== "" && status !== RuntimeLoader.Error
            position: Qt.vector3d(root.egoX, root.egoY, 0)
            scale: Qt.vector3d(root.egoScale, root.egoScale, root.egoScale)
            eulerRotation: Qt.vector3d(root.egoRotX, root.egoRotY + root.egoYawDeg, root.egoRotZ)
        }
        Timer {
            id: egoFitTimer
            interval: 33
            repeat: true
            running: egoCar.status === RuntimeLoader.Success && !root.egoFitted
            onTriggered: root.fitEgoFromBounds()
        }
    }

    MouseArea {
        id: orbitArea
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.top: parent.top
        anchors.bottom: toolbar.top
        acceptedButtons: Qt.LeftButton | Qt.RightButton
        hoverEnabled: true
        property real lastX: 0
        property real lastY: 0

        onPressed: function (mouse) {
            lastX = mouse.x
            lastY = mouse.y
        }
        onPositionChanged: function (mouse) {
            const dx = mouse.x - lastX
            const dy = mouse.y - lastY
            lastX = mouse.x
            lastY = mouse.y
            if (mouse.buttons & Qt.LeftButton) {
                // Full 360° yaw (painter BEV used ±85° because fake perspective broke).
                root.yawDeg = root.yawDeg + dx * 0.35
                root.pitchDeg = Math.max(8, Math.min(85, root.pitchDeg - dy * 0.35))
            } else if (mouse.buttons & Qt.RightButton) {
                root.panX += dx * 0.03
                root.panY -= dy * 0.03
            }
        }
    }

    WheelHandler {
        onWheel: function (event) {
            const f = event.angleDelta.y > 0 ? 1.12 : 0.88
            const z = root.zoomFactor * f
            if (z >= 0.4 && z <= 4.5)
                root.zoomFactor = z
            event.accepted = true
        }
    }

    // Per-object metric readout: prints each rendered car's (x, z) in metres
    // beside its model, so BEV placement can be checked against the
    // range_diagnostics.py CSV without instrumenting the renderer.
    Repeater {
        model: root.debugMetric ? root.debugRows : []
        delegate: Text {
            // mapFrom3DScene projects the metric scene point into viewport px,
            // so the label tracks the car under camera orbit and zoom.
            readonly property vector3d vp: view.mapFrom3DScene(
                Qt.vector3d(modelData.posX, 1.2, modelData.posZ))
            visible: vp.z > 0
            x: vp.x + 8
            y: vp.y - 8
            z: 4
            text: "#" + modelData.tid + "  x=" + modelData.posX.toFixed(2)
                  + "  z=" + (-modelData.posZ).toFixed(1) + " m"
            color: "#ffd60a"
            font.pixelSize: 10
            font.bold: true
            style: Text.Outline
            styleColor: "#000000"
        }
    }

    Column {
        anchors.left: parent.left
        anchors.top: parent.top
        anchors.margins: 8
        spacing: 4
        z: 3

        Text {
            visible: root.debugMetric
            text: "DEBUG METRIC  centreline x=0, ticks 10 m"
            color: "#ff9500"
            font.pixelSize: 10
            font.bold: true
        }
        Text {
            text: "QT QUICK 3D BEV"
            color: "#c9d1d9"
            font.pixelSize: 11
            font.bold: true
        }
        Rectangle {
            width: cipoBadgeTxt.implicitWidth + 16
            height: 18
            radius: 3
            color: root.cipoStatus === "DANGER" ? "#da3633"
                 : (root.cipoStatus === "WARNING" ? "#d9822b"
                 : (root.cipoStatus === "DEGRADED" ? "#6e7681" : "#238636"))
            Text {
                id: cipoBadgeTxt
                anchors.centerIn: parent
                text: root.cipoVisible
                      ? ("CIPO: " + root.cipoStatus + "  " + root.cipoDist.toFixed(1) + "m")
                      : ("CIPO: " + root.cipoStatus)
                color: "#ffffff"
                font.pixelSize: 9
                font.bold: true
            }
        }
        Text {
            text: egoCar.status === RuntimeLoader.Error
                  ? ("GLB error: " + egoCar.errorString)
                  : (egoCar.status === RuntimeLoader.Success
                     ? (root.laneHeld ? "lane held — dead reckoning"
                        : (root.laneValid
                           ? "lane locked — ego centered"
                           : "no lane"))
                     : root.overlayHint)
            color: egoCar.status === RuntimeLoader.Error ? "#f85149"
                 : (root.laneHeld ? "#d9822b" : "#8b949e")
            font.pixelSize: 9
        }
    }

    Rectangle {
        id: toolbar
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        height: 22
        color: "#b0161b22"
        z: 3

        Row {
            anchors.verticalCenter: parent.verticalCenter
            anchors.left: parent.left
            anchors.leftMargin: 6
            spacing: 6

            Repeater {
                model: [
                    { key: "reset", label: "Reset" },
                    { key: "scenic", label: root.scenicView ? "Scene" : "Road" },
                    { key: "road", label: root.cinematicRoad ? "Film" : "Grid" },
                    { key: "lanes", label: root.showLaneLines ? "Lanes" : "No lanes" },
                    { key: "env", label: root.envLabel },
                    { key: "cal", label: "Cal" }
                ]
                Rectangle {
                    required property var modelData
                    width: chipText.implicitWidth + 10
                    height: 16
                    radius: 3
                    color: chipMouse.containsMouse ? "#30363d" : "#21262d"
                    Text {
                        id: chipText
                        anchors.centerIn: parent
                        text: parent.modelData.label
                        color: "#c9d1d9"
                        font.pixelSize: 9
                    }
                    MouseArea {
                        id: chipMouse
                        anchors.fill: parent
                        hoverEnabled: true
                        onClicked: {
                            const key = parent.modelData.key
                            if (key === "reset") {
                                root.pitchDeg = 13
                                root.yawDeg = 0
                                root.zoomFactor = 1.08
                                root.panX = 0
                                root.panY = 0
                                root.calibPitch = 0
                                root.calibH = 1
                            } else if (key === "scenic") {
                                root.scenicView = !root.scenicView
                            } else if (key === "road") {
                                root.cinematicRoad = !root.cinematicRoad
                            } else if (key === "env") {
                                root.cycleEnvMode()
                            } else if (key === "cal") {
                                root.showCalib = !root.showCalib
                            } else {
                                root.showLaneLines = !root.showLaneLines
                            }
                        }
                    }
                }
            }

            Text {
                visible: root.showCalib
                text: "P " + root.calibPitch.toFixed(0)
                color: "#8b949e"
                font.pixelSize: 9
            }
            Rectangle {
                visible: root.showCalib
                width: 70
                height: 5
                radius: 2
                color: "#2d3440"
                Rectangle {
                    width: 8
                    height: 8
                    radius: 4
                    color: "#58a6ff"
                    y: -1.5
                    x: Math.max(0, Math.min(parent.width - 8,
                       (root.calibPitch + 10) / 20 * (parent.width - 8)))
                    MouseArea {
                        anchors.fill: parent
                        anchors.margins: -6
                        drag.target: parent
                        drag.axis: Drag.XAxis
                        drag.minimumX: 0
                        drag.maximumX: 62
                        onPositionChanged: {
                            parent.x = Math.max(0, Math.min(62, parent.x))
                            root.calibPitch = -10 + (parent.x / 62) * 20
                        }
                    }
                }
            }
            Text {
                visible: root.showCalib
                text: "H " + root.calibH.toFixed(1)
                color: "#8b949e"
                font.pixelSize: 9
            }
            Rectangle {
                visible: root.showCalib
                width: 54
                height: 5
                radius: 2
                color: "#2d3440"
                Rectangle {
                    width: 8
                    height: 8
                    radius: 4
                    color: "#58a6ff"
                    y: -1.5
                    x: Math.max(0, Math.min(parent.width - 8,
                       (root.calibH - 1.0) / 2.0 * (parent.width - 8)))
                    MouseArea {
                        anchors.fill: parent
                        anchors.margins: -6
                        drag.target: parent
                        drag.axis: Drag.XAxis
                        drag.minimumX: 0
                        drag.maximumX: 46
                        onPositionChanged: {
                            parent.x = Math.max(0, Math.min(46, parent.x))
                            root.calibH = 1.0 + (parent.x / 46) * 2.0
                        }
                    }
                }
            }
        }
    }
}
