/* =========================================
   PITCHVISION SYSTEM VISUALIZER
========================================= */


const stages = {

    1: {

        title: "Object Detection",

        status: "DETECTION",

        number: "01",

        description:
            "The detection stage identifies football-related objects and returns their locations, classes and confidence scores.",

        objects: "9",

        teams: "—",

        formation: "—",

        position: "IMAGE"

    },


    2: {

        title: "Team Classification",

        status: "CLASSIFICATION",

        number: "02",

        description:
            "Detected players are separated into their respective teams using visual appearance and jersey information.",

        objects: "Players",

        teams: "2",

        formation: "—",

        position: "CLASSIFIED"

    },


    3: {

        title: "Spatial Mapping",

        status: "MAPPING",

        number: "03",

        description:
            "Player locations are transformed from the camera image into normalized coordinates on a top-down pitch.",

        objects: "Players",

        teams: "2",

        formation: "—",

        position: "MAPPED"

    },


    4: {

        title: "Formation Analysis",

        status: "FORMATION",

        number: "04",

        description:
            "The spatial distribution of players is analyzed to estimate the structure and formation of each team.",

        objects: "22",

        teams: "2",

        formation: "4-3-3",

        position: "STRUCTURED"

    },


    5: {

        title: "Tactical Insights",

        status: "ANALYSIS",

        number: "05",

        description:
            "Player distributions are used to calculate basic tactical characteristics such as width, compactness and positional density.",

        objects: "22",

        teams: "2",

        formation: "4-3-3",

        position: "ANALYZED"

    }

};


/* =========================================
   ELEMENTS
========================================= */

const stageTitle =
    document.getElementById(
        "stageTitle"
    );


const stageNumber =
    document.getElementById(
        "stageNumber"
    );


const panelStatus =
    document.getElementById(
        "panelStatus"
    );


const outputNumber =
    document.getElementById(
        "outputNumber"
    );


const outputTitle =
    document.getElementById(
        "outputTitle"
    );


const outputText =
    document.getElementById(
        "outputText"
    );


const objects =
    document.getElementById(
        "objects"
    );


const teams =
    document.getElementById(
        "teams"
    );


const formation =
    document.getElementById(
        "formation"
    );


const position =
    document.getElementById(
        "position"
    );


const nextButton =
    document.getElementById(
        "nextStage"
    );


const previousButton =
    document.getElementById(
        "previousStage"
    );


const stageButtons =
    document.querySelectorAll(
        ".stage-button"
    );


const pipelineSteps =
    document.querySelectorAll(
        ".pipeline-step"
    );


const detectionBoxes =
    document.querySelectorAll(
        ".yolo-box"
    );


let currentStage = 1;


/* =========================================
   UPDATE VISUALIZER
========================================= */

function updateVisualizer(stage) {

    currentStage = stage;


    const data =
        stages[stage];


    /*
        Header
    */

    stageTitle.textContent =
        data.title;


    stageNumber.textContent =
        data.number;


    panelStatus.textContent =
        data.status;


    /*
        Output
    */

    outputNumber.textContent =
        data.number;


    outputTitle.textContent =
        data.title;


    outputText.textContent =
        data.description;


    /*
        Metrics
    */

    objects.textContent =
        data.objects;


    teams.textContent =
        data.teams;


    formation.textContent =
        data.formation;


    position.textContent =
        data.position;


    /*
        Stage buttons
    */

    stageButtons.forEach(
        button => {

            const buttonStage =
                Number(
                    button.dataset.stage
                );


            button.classList.toggle(
                "active",
                buttonStage === stage
            );

        }
    );


    /*
        Pipeline
    */

    pipelineSteps.forEach(
        step => {

            const stepStage =
                Number(
                    step.dataset.stage
                );


            step.classList.toggle(
                "active",
                stepStage === stage
            );

        }
    );


    /*
        Detection boxes

        They are most important
        during the detection stage.
    */

    if (stage === 1) {

        detectionBoxes.forEach(
            box => {

                box.style.opacity = "1";

            }
        );

    }

    else if (stage === 2) {

        detectionBoxes.forEach(
            box => {

                box.style.opacity = "0.65";

            }
        );

    }

    else {

        detectionBoxes.forEach(
            box => {

                box.style.opacity = "0.18";

            }
        );

    }


    /*
        Tactical pitch

        Make the tactical representation
        stronger as the pipeline progresses.
    */

    const tacticalView =
        document.querySelector(
            ".tactical-view"
        );


    if (stage < 3) {

        tacticalView.style.opacity =
            "0.45";

    }

    else {

        tacticalView.style.opacity =
            "1";

    }

}


/* =========================================
   NEXT
========================================= */

nextButton.addEventListener(
    "click",
    () => {

        if (currentStage < 5) {

            updateVisualizer(
                currentStage + 1
            );

        }

    }
);


/* =========================================
   PREVIOUS
========================================= */

previousButton.addEventListener(
    "click",
    () => {

        if (currentStage > 1) {

            updateVisualizer(
                currentStage - 1
            );

        }

    }
);


/* =========================================
   STAGE BUTTONS
========================================= */

stageButtons.forEach(
    button => {

        button.addEventListener(
            "click",
            () => {

                const stage =
                    Number(
                        button.dataset.stage
                    );


                updateVisualizer(
                    stage
                );

            }
        );

    }
);


/* =========================================
   PIPELINE CLICK
========================================= */

pipelineSteps.forEach(
    step => {

        step.addEventListener(
            "click",
            () => {

                const stage =
                    Number(
                        step.dataset.stage
                    );


                updateVisualizer(
                    stage
                );


                document
                    .getElementById("demo")
                    .scrollIntoView({
                        behavior: "smooth"
                    });

            }
        );

    }
);


/* =========================================
   KEYBOARD NAVIGATION
========================================= */

document.addEventListener(
    "keydown",
    event => {

        if (
            event.key === "ArrowRight"
            &&
            currentStage < 5
        ) {

            updateVisualizer(
                currentStage + 1
            );

        }


        if (
            event.key === "ArrowLeft"
            &&
            currentStage > 1
        ) {

            updateVisualizer(
                currentStage - 1
            );

        }

    }
);


/* =========================================
   INITIAL STATE
========================================= */

updateVisualizer(1);