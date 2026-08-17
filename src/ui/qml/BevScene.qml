import QtQuick
import QtQuick3D
import QtQuick3D.AssetUtils

Item {
    id: root

    // Camera — same defaults as BEVWidget (chase cam)
    property real pitchDeg: 31.0
    property real yawDeg: 0.0
    property real zoomFactor: 1.05
    property real camH: 10.0
    property real camDist: 12.0
    property real calibPitch: -7.0
    property real calibH: 1.0
    property real panX: 0.0
    property real panY: 0.0
    property bool cinematicRoad: true
    property bool showLaneLines: true
    property bool showCalib: false
    property string cipoStatus: "SAFE"
    property bool cipoVisible: false
    property real cipoX: 0
    property real cipoZ: -12
    property real cipoDist: 0
    readonly property color cipoGlow: cipoStatus === "DANGER" ? "#e23a3c"
                                     : (cipoStatus === "WARNING" ? "#e09a20" : "#2ecc71")
    readonly property color corridorColor: cipoStatus === "DANGER" ? Qt.rgba(0.86, 0.18, 0.18, 0.42)
                                     : (cipoStatus === "WARNING" ? Qt.rgba(0.86, 0.58, 0.08, 0.40)
                                                                 : Qt.rgba(0.18, 0.86, 0.44, 0.38))
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
    property real targetCarLength: 4.6
    property bool egoFitted: false
    property string egoDebug: "bounds pending"
    property string trafficJson: "[]"
    property int trafficCount: 0
    property string corridorJson: "[]"
    property string laneJson: "[]"
    property string dashJson: "[]"

    onTrafficJsonChanged: applyTraffic()
    onTeslaYChanged: applyTraffic()
    onCorridorJsonChanged: applySegPool(corrRep, root.corridorJson, 0.035)
    onLaneJsonChanged: applyLanes()
    onDashJsonChanged: applyDashes()
    onShowLaneLinesChanged: { applyLanes(); applyDashes() }
    onCinematicRoadChanged: { applyLanes(); applyDashes() }

    Component.onCompleted: {
        applySegPool(corrRep, root.corridorJson, 0.035)
        applyLanes()
        applyDashes()
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

    function applyTraffic() {
        let rows = []
        try { rows = JSON.parse(root.trafficJson || "[]") } catch (e) { rows = [] }
        const pools = [
            [skoda0, skoda1, skoda2],
            [tesla0],
            [shc0],
            [truck0]
        ]
        const used = [0, 0, 0, 0]
        for (let i = 0; i < rows.length; i++) {
            const r = rows[i]
            let kind = r.kind | 0
            if (kind < 0 || kind > 3)
                kind = 0
            if (used[kind] >= pools[kind].length)
                kind = 0
            if (used[kind] >= pools[kind].length)
                continue
            const node = pools[kind][used[kind]++]
            node.visible = true
            const py = (kind === 1) ? root.teslaY : r.posY
            node.position = Qt.vector3d(r.posX, py, r.posZ)
            // Same heading as ego (rear toward chase cam). Tesla mesh X-90 is on teslaFix.
            node.eulerRotation = Qt.vector3d(0, root.egoRotY, 0)
        }
        for (let k = 0; k < pools.length; k++) {
            for (let j = used[k]; j < pools[k].length; j++)
                pools[k][j].visible = false
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
            root.skodaScale = s
            root.egoY = -mn.y * s
            root.egoDebug += "  → scale " + s.toFixed(1)
            // Tesla/SHC/Dodge keep their own GLB node scales — do not copy Audi fit.
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
        camera: sceneCamera

        environment: SceneEnvironment {
            backgroundMode: SceneEnvironment.Color
            clearColor: "#0e1218"
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
            fieldOfView: 50
            clipNear: 0.3
            clipFar: 250
        }

        DirectionalLight {
            eulerRotation.x: -42
            eulerRotation.y: 28
            brightness: 1.35
            color: "#fff4e8"
            castsShadow: false
            ambientColor: "#4a5564"
        }

        // Ground: cinematic asphalt vs telemetry grid
        Model {
            source: "#Rectangle"
            eulerRotation.x: -90
            scale: Qt.vector3d(root.cinematicRoad ? 0.112 : 0.80, root.cinematicRoad ? 0.85 : 1.60, 1)
            position: Qt.vector3d(0, 0, root.cinematicRoad ? -40 : 0)
            materials: PrincipledMaterial {
                baseColor: root.cinematicRoad ? "#101216" : "#222a35"
                roughness: 0.95
                metalness: 0.0
            }
        }

        // Shoulder beyond cinematic asphalt
        Model {
            visible: root.cinematicRoad
            source: "#Rectangle"
            eulerRotation.x: -90
            scale: Qt.vector3d(0.28, 0.85, 1)
            position: Qt.vector3d(0, -0.01, -40)
            materials: PrincipledMaterial {
                baseColor: "#0a0c10"
                roughness: 1.0
            }
        }

        Model {
            visible: root.cinematicRoad
            source: "#Cube"
            position: Qt.vector3d(-5.35, 0.025, -40)
            scale: Qt.vector3d(0.0012, 0.0002, 0.78)
            materials: PrincipledMaterial {
                lighting: PrincipledMaterial.NoLighting
                baseColor: "#ebefff"
            }
        }
        Model {
            visible: root.cinematicRoad
            source: "#Cube"
            position: Qt.vector3d(5.35, 0.025, -40)
            scale: Qt.vector3d(0.0012, 0.0002, 0.78)
            materials: PrincipledMaterial {
                lighting: PrincipledMaterial.NoLighting
                baseColor: "#ebefff"
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
                    baseColor: "#e8eef8"
                }
            }
        }

        Repeater3D {
            model: 17
            visible: !root.cinematicRoad
            Model {
                source: "#Cube"
                position: Qt.vector3d((index - 8) * 5.0, 0.01, -40)
                scale: Qt.vector3d(0.00012, 0.00008, 1.20)
                materials: PrincipledMaterial {
                    baseColor: "#2a3340"
                    roughness: 1
                }
            }
        }
        Repeater3D {
            model: 17
            visible: !root.cinematicRoad
            Model {
                source: "#Cube"
                position: Qt.vector3d(0, 0.01, -index * 5.0)
                scale: Qt.vector3d(0.80, 0.00008, 0.00012)
                materials: PrincipledMaterial {
                    baseColor: "#2a3340"
                    roughness: 1
                }
            }
        }

        Repeater3D {
            id: corrRep
            model: 14
            Model {
                visible: false
                source: "#Cube"
                materials: PrincipledMaterial {
                    lighting: PrincipledMaterial.NoLighting
                    baseColor: root.corridorColor
                    opacity: 0.45
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
                    baseColor: pal === 1 ? "#ffbe00" : (pal === 2 ? "#00ffb4" : "#78dcff")
                }
            }
        }

        Model {
            id: cipoRing
            visible: root.cipoVisible
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
            visible: root.cipoVisible
            source: "#Cylinder"
            position: Qt.vector3d(root.cipoX, 1.35, root.cipoZ)
            scale: Qt.vector3d(0.0014, 0.026, 0.0014)
            materials: PrincipledMaterial {
                lighting: PrincipledMaterial.NoLighting
                baseColor: root.cipoGlow
            }
        }
        Node {
            visible: root.cipoVisible
            position: Qt.vector3d(root.cipoX, 2.55, root.cipoZ)
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

        // Placeholder cube if the GLB is missing or failed
        Model {
            visible: root.egoGltf.toString() === "" || egoCar.status === RuntimeLoader.Error
            source: "#Cube"
            position: Qt.vector3d(0, 0.65, 0)
            scale: Qt.vector3d(0.018, 0.013, 0.043)
            materials: PrincipledMaterial {
                baseColor: "#00c8ff"
                roughness: 0.35
                metalness: 0.15
            }
        }

        // Model.source only loads Qt .mesh (balsam). GLB must use RuntimeLoader.
        RuntimeLoader {
            id: egoCar
            source: root.egoGltf
            visible: root.egoGltf.toString() !== "" && status !== RuntimeLoader.Error
            position: Qt.vector3d(0, root.egoY, 0)
            scale: Qt.vector3d(root.egoScale, root.egoScale, root.egoScale)
            eulerRotation: Qt.vector3d(root.egoRotX, root.egoRotY, root.egoRotZ)
        }

        Timer {
            id: egoFitTimer
            interval: 33
            repeat: true
            running: egoCar.status === RuntimeLoader.Success && !root.egoFitted
            onTriggered: root.fitEgoFromBounds()
        }

        // Pooled traffic: one RuntimeLoader per slot, source set once.
        Node {
            id: skoda0
            visible: false
            RuntimeLoader {
                source: root.skodaGltf
                scale: Qt.vector3d(root.skodaScale, root.skodaScale, root.skodaScale)
            }
        }
        Node {
            id: skoda1
            visible: false
            RuntimeLoader {
                source: root.skodaGltf
                scale: Qt.vector3d(root.skodaScale, root.skodaScale, root.skodaScale)
            }
        }
        Node {
            id: skoda2
            visible: false
            RuntimeLoader {
                source: root.skodaGltf
                scale: Qt.vector3d(root.skodaScale, root.skodaScale, root.skodaScale)
            }
        }
        Node {
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
        Node {
            id: shc0
            visible: false
            RuntimeLoader {
                source: root.shcGltf
                scale: Qt.vector3d(root.shcScale, root.shcScale, root.shcScale)
                eulerRotation: Qt.vector3d(0, root.shcRotY, 0)
            }
        }
        Node {
            id: truck0
            visible: false
            RuntimeLoader {
                source: root.dodgeGltf
                scale: Qt.vector3d(root.dodgeScale, root.dodgeScale, root.dodgeScale)
            }
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

    Column {
        anchors.left: parent.left
        anchors.top: parent.top
        anchors.margins: 8
        spacing: 4

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
                 : (root.cipoStatus === "WARNING" ? "#d9822b" : "#238636")
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
                     ? ("Phase 4 — ego + " + root.trafficCount + " traffic")
                     : root.overlayHint)
            color: egoCar.status === RuntimeLoader.Error ? "#f85149" : "#8b949e"
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

        Row {
            anchors.verticalCenter: parent.verticalCenter
            anchors.left: parent.left
            anchors.leftMargin: 6
            spacing: 6

            Repeater {
                model: [
                    { key: "reset", label: "Reset" },
                    { key: "road", label: root.cinematicRoad ? "Film" : "Grid" },
                    { key: "lanes", label: root.showLaneLines ? "Lanes" : "No lanes" },
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
                                root.pitchDeg = 31
                                root.yawDeg = 0
                                root.zoomFactor = 1.05
                                root.panX = 0
                                root.panY = 0
                                root.calibPitch = -7
                                root.calibH = 1
                            } else if (key === "road") {
                                root.cinematicRoad = !root.cinematicRoad
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
