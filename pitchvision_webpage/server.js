const express = require("express");
const path = require("path");

const app = express();
const PORT = 3000;

// Serve frontend
app.use(express.static(path.join(__dirname, "public")));

app.use(express.json());

// Demo API
app.get("/api/status", (req, res) => {
    res.json({
        project: "PitchVision",
        status: "online",
        pipeline: [
            "Detection",
            "Team Classification",
            "Pitch Mapping",
            "Formation Analysis",
            "Tactical Insights"
        ]
    });
});

app.listen(PORT, () => {
    console.log(`PitchVision running at http://localhost:${PORT}`);
});