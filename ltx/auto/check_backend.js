/**
 * Automated Video Generation Job Runner for LTX-2.3 Backend API.
 * Uploads reference image and audio files directly to backend.py,
 * submits jobs to the queue, and polls real-time progress until complete.
 */

const fs = require("node:fs");
const path = require("node:path");

// Configure the backend service URL (override with BACKEND_URL env var if using Cloudflare tunnel)
const BACKEND_URL = (process.env.BACKEND_URL || "http://127.0.0.1:8000").replace(/\/+$/, "");
const POLL_INTERVAL_MS = 3000;

// Add jobs here. They will run sequentially on the backend queue.
const jobs = [
    {
        prompt: "Use the supplied image as the visual reference and the supplied audio as the exact audio source. Generate a photorealistic video with accurate lip synchronization and natural facial movements. Preserve the character's identity and appearance.",
        image: "./input/1.webp",
        audio: "./input/1.wav",
        seed: -1,
        duration: "5 Seconds (121 frames)",
        resolution: "480p",
        aspectRatio: "Auto (match reference image)",
        matchAudioDuration: false
    },

    // Add more jobs by copying this block and changing the paths and prompt:
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

/**
 * Check connectivity and retrieve health status from backend.py.
 */
async function checkBackendHealth() {
    log(`Checking backend connection at ${BACKEND_URL} ...`);
    try {
        const res = await fetch(`${BACKEND_URL}/api/health`);
        if (!res.ok) {
            throw new Error(`HTTP error ${res.status}: ${res.statusText}`);
        }
        const data = await res.json();
        log(`Connected to backend. Status: ${data.status} | Model Loaded: ${data.model_loaded}`);
        if (data.gpus && data.gpus.length > 0) {
            for (const gpu of data.gpus) {
                log(`  GPU ${gpu.index} (${gpu.name}): ${gpu.free_vram_gb} GB / ${gpu.total_vram_gb} GB free`);
            }
        }
        return true;
    } catch (err) {
        log(`⚠️ Could not reach backend at ${BACKEND_URL}: ${err.message}`);
        log(`Make sure backend.py is running: python backend.py --port 8000`);
        return false;
    }
}

/**
 * Uploads media assets and registers a video generation job on backend.py.
 */
async function uploadAndLaunchJob(job, jobNumber) {
    // Resolve file paths relative to script location or current directory
    const imagePath = path.isAbsolute(job.image) ? job.image : path.resolve(__dirname, job.image);
    const audioPath = path.isAbsolute(job.audio) ? job.audio : path.resolve(__dirname, job.audio);

    if (!fs.existsSync(imagePath)) {
        throw new Error(`Image file not found: ${imagePath}`);
    }
    if (!fs.existsSync(audioPath)) {
        throw new Error(`Audio file not found: ${audioPath}`);
    }

    log(`Reading files for Job ${jobNumber}...`);
    const imageBuffer = fs.readFileSync(imagePath);
    const audioBuffer = fs.readFileSync(audioPath);

    const formData = new FormData();
    formData.append("prompt", job.prompt);
    formData.append("seed", String(job.seed ?? -1));
    formData.append("duration", job.duration ?? "5 Seconds (121 frames)");
    formData.append("resolution", job.resolution ?? "480p");
    formData.append("aspect_ratio", job.aspectRatio ?? "Auto (match reference image)");
    formData.append("match_audio_dur", String(job.matchAudioDuration ?? false));

    // Attach binary files
    const imageBlob = new Blob([imageBuffer], { type: "image/webp" });
    const audioBlob = new Blob([audioBuffer], { type: "audio/wav" });
    formData.append("image_file", imageBlob, path.basename(imagePath));
    formData.append("audio_file", audioBlob, path.basename(audioPath));

    log(`Submitting Job ${jobNumber} to ${BACKEND_URL}/api/generate/upload ...`);
    const response = await fetch(`${BACKEND_URL}/api/generate/upload`, {
        method: "POST",
        body: formData
    });

    if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Upload failed (${response.status}): ${errorText}`);
    }

    const result = await response.json();
    log(`Job ${jobNumber} registered with ID: ${result.job_id} (Status: ${result.status})`);
    return result.job_id;
}

/**
 * Polls the job status until completed, failed, or cancelled.
 */
async function pollJobStatus(jobId, jobNumber) {
    let lastProgress = -1;
    let lastDesc = "";
    let lastStatus = "";

    while (true) {
        try {
            const res = await fetch(`${BACKEND_URL}/api/jobs/${jobId}`);
            if (!res.ok) {
                throw new Error(`Failed to fetch job status: HTTP ${res.status}`);
            }

            const job = await res.json();
            const currentStatus = job.status;
            const progress = job.progress || 0;
            const desc = job.progress_desc || "";

            // Log status transitions
            if (currentStatus !== lastStatus) {
                log(`Job ${jobNumber} [${jobId}] status -> ${currentStatus.toUpperCase()}`);
                lastStatus = currentStatus;
            }

            // Log progress updates
            if (progress !== lastProgress || desc !== lastDesc) {
                const pct = Math.round(progress * 100);
                const descStr = desc ? ` (${desc})` : "";
                log(`Job ${jobNumber} progress: ${pct}%${descStr}`);
                lastProgress = progress;
                lastDesc = desc;
            }

            // Check completion conditions
            if (currentStatus === "completed") {
                log(`✅ Job ${jobNumber} completed in ${job.execution_time ? job.execution_time.toFixed(1) + 's' : 'unknown time'}!`);
                if (job.result_video_path) {
                    log(`📁 Output file: ${job.result_video_path}`);
                }
                if (job.video_url) {
                    log(`🌐 Video URL: ${BACKEND_URL}${job.video_url}`);
                }
                return job;
            }

            if (currentStatus === "failed") {
                throw new Error(`Job execution failed: ${job.error || "Unknown error"}`);
            }

            if (currentStatus === "cancelled") {
                throw new Error("Job was cancelled");
            }

        } catch (err) {
            // If network hiccup during polling, log and keep retrying
            if (!err.message.includes("failed:") && !err.message.includes("cancelled")) {
                log(`⚠️ Polling notice: ${err.message}`);
            } else {
                throw err;
            }
        }

        await sleep(POLL_INTERVAL_MS);
    }
}

async function main() {
    console.log("========================================");
    console.log("🎬 LTX Video Generator - Backend API Client");
    console.log(`📡 Backend URL: ${BACKEND_URL}`);
    console.log(`📋 Total Jobs in queue: ${jobs.length}`);
    console.log("========================================\n");

    const isHealthy = await checkBackendHealth();
    if (!isHealthy) {
        console.error("\n❌ Exiting: Backend service is not reachable.");
        process.exitCode = 1;
        return;
    }

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
        log(`Duration: ${job.duration ?? "5s"} | Res: ${job.resolution ?? "480p"}`);
        console.log("========================================");

        try {
            const jobId = await uploadAndLaunchJob(job, jobNumber);
            const finishedJob = await pollJobStatus(jobId, jobNumber);
            completed++;
        } catch (error) {
            failed++;
            console.error(`\n[${timestamp()}] ❌ Job ${jobNumber} failed:`, error.message || error);
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
