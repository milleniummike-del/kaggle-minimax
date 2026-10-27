
const fs = require("node:fs");
const path = require("node:path");

const GRADIO_URL = "https://e91864d9661c0a8b81.gradio.live";
const POLL_INTERVAL_MS = 3000;

// Add jobs here. They will run sequentially, one video at a time.
const jobs = [
    {
        prompt: "Use the supplied image as the visual reference and the supplied audio as the exact audio source. Generate a photorealistic video with accurate lip synchronization and natural facial movements. Preserve the character's identity and appearance.",
        image: "./input/1.webp",
        audio: "./input/1.wav",
        seed: -1,
        duration: "5 Seconds (121 frames)",
        resolution: "420p",
        aspectRatio: "Auto (match reference image)",
        matchAudioDuration: false
    },

    // Add more jobs by copying this block and changing the paths and prompt.
    /*
    {
        prompt: "Generate a cinematic video using the supplied image and audio.",
        image: "./input/2.webp",
        audio: "./input/2.wav",
        seed: -1,
        duration: "5 Seconds (121 frames)",
        resolution: "720p",
        aspectRatio: "Auto (match reference image)",
        matchAudioDuration: true
    }
    */
];

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

function timestamp() {
    return new Date().toLocaleTimeString();
}

function log(message) {
    console.log(`[${timestamp()}] ${message}`);
}

function formatStatus(data) {
    if (data === undefined || data === null) return "No status returned";
    if (typeof data === "string") return data;
    return JSON.stringify(data);
}

async function main() {
    const { Client, handle_file } = await import("@gradio/client");

    log("Connecting to Gradio...");
    const app = await Client.connect(GRADIO_URL);
    log(`Connected. ${jobs.length} job(s) in queue.`);

    let completed = 0;
    let failed = 0;

    for (let index = 0; index < jobs.length; index++) {
        const job = jobs[index];
        const jobNumber = index + 1;

        console.log("\n========================================");
        log(`Starting job ${jobNumber}/${jobs.length}`);
        log(`Prompt: ${job.prompt}`);
        log(`Image: ${job.image}`);
        log(`Audio: ${job.audio}`);
        console.log("========================================");

        try {
            const imagePath = path.resolve(job.image);
            const audioPath = path.resolve(job.audio);

            if (!fs.existsSync(imagePath)) {
                throw new Error(`Image file not found: ${imagePath}`);
            }

            if (!fs.existsSync(audioPath)) {
                throw new Error(`Audio file not found: ${audioPath}`);
            }

            log("Uploading image and audio...");
            const imageFile = await handle_file(imagePath);
            const audioFile = await handle_file(audioPath);

            const args = [
                job.prompt,
                imageFile,
                audioFile,
                job.seed ?? -1,
                job.duration ?? "5 Seconds (121 frames)",
                job.resolution ?? "480p",
                job.aspectRatio ?? "Auto (match reference image)",
                job.matchAudioDuration ?? true,
                null
            ];

            // Start generation without blocking the status-monitoring loop.
            let settled = false;

            const generationPromise = app.predict("/generate_video", args).then(
                result => {
                    settled = true;
                    return result;
                },
                error => {
                    settled = true;
                    throw error;
                }
            );

            let previousStatus = "";

            while (!settled) {
                try {
                    const statusResult = await app.predict(
                        "/get_queue_status",
                        []
                    );

                    const status = formatStatus(statusResult.data);

                    // Avoid printing the same status repeatedly.
                    if (status !== previousStatus) {
                        log(`Queue status: ${status}`);
                        previousStatus = status;
                    }
                } catch (statusError) {
                    log(`Status polling unavailable: ${statusError.message}`);
                }

                if (!settled) {
                    await Promise.race([
                        sleep(POLL_INTERVAL_MS),
                        generationPromise.then(
                            () => undefined,
                            () => undefined
                        )
                    ]);
                }
            }

            const result = await generationPromise;

            log(`Job ${jobNumber} generation returned.`);
            console.dir(result.data, { depth: null });

            completed++;

        } catch (error) {
            failed++;
            console.error(
                `\n[${timestamp()}] Job ${jobNumber} failed:`,
                error.message || error
            );

            // Continue to the next job instead of stopping the queue.
        }

        log(`Queue progress: ${index + 1}/${jobs.length} processed; ${completed} succeeded; ${failed} failed.`);
    }

    console.log("\n========================================");
    log("ALL QUEUED JOBS PROCESSED");
    log(`Successful: ${completed}`);
    log(`Failed: ${failed}`);
    console.log("========================================");
}

main().catch(error => {
    console.error("Fatal error:", error);
    process.exitCode = 1;
});
