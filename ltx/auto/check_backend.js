/**
 * Automated Video Generation Batch Runner for LTX-2.3 Backend API.
 * Reads a project folder containing `manifest.json`, loops through each piece,
 * uploads corresponding audio (.wav) and reference image (.webp/.png) assets,
 * tracks real-time progress on backend.py, and downloads the finished videos.
 *
 * Usage:
 *   node auto/check_backend.js
 *   node auto/check_backend.js "X:\inputmixed\Copper Thunder_split_pieces_27_parts (1)"
 *   node auto/check_backend.js --folder "path/to/folder" --start 5 --end 10 --res 720p
 */

const fs = require("node:fs");
const path = require("node:path");

// ---- Configuration Defaults ----
const DEFAULT_FOLDER = "X:\\inputmixed\\All Was Good_split_pieces_35_parts";
const DEFAULT_BACKEND_URL = "https://summit-alex-theory-neither.trycloudflare.com";
const POLL_INTERVAL_MS = 3000;

// Parse CLI Arguments
function parseCliArgs() {
    const args = process.argv.slice(2);
    const config = {
        folder: process.env.INPUT_DIR || process.env.INPUT_FOLDER || null,
        backendUrl: process.env.BACKEND_URL || DEFAULT_BACKEND_URL,
        resolution: process.env.RESOLUTION || "240P",
        aspectRatio: process.env.ASPECT_RATIO || "Auto (match reference image)",
        matchAudioDuration: true,
        startIndex: 1,
        endIndex: null,
        singlePiece: null,
        downloadVideos: true,
        skipExisting: true,
        seed: -1,
    };

    for (let i = 0; i < args.length; i++) {
        const arg = args[i];
        if ((arg === "--folder" || arg === "-f") && args[i + 1]) {
            config.folder = args[++i];
        } else if ((arg === "--url" || arg === "-u") && args[i + 1]) {
            config.backendUrl = args[++i];
        } else if ((arg === "--res" || arg === "--resolution") && args[i + 1]) {
            config.resolution = args[++i];
        } else if ((arg === "--start" || arg === "--from") && args[i + 1]) {
            config.startIndex = parseInt(args[++i], 10);
        } else if ((arg === "--end" || arg === "--to") && args[i + 1]) {
            config.endIndex = parseInt(args[++i], 10);
        } else if (arg === "--piece" && args[i + 1]) {
            config.singlePiece = parseInt(args[++i], 10);
        } else if (arg === "--seed" && args[i + 1]) {
            config.seed = parseInt(args[++i], 10);
        } else if (arg === "--no-download") {
            config.downloadVideos = false;
        } else if (arg === "--no-skip" || arg === "--force") {
            config.skipExisting = false;
        } else if (!arg.startsWith("-") && !config.folder) {
            config.folder = arg;
        }
    }

    if (!config.folder) {
        config.folder = DEFAULT_FOLDER;
    }

    config.backendUrl = config.backendUrl.replace(/\/+$/, "");
    return config;
}

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

function timestamp() {
    return new Date().toLocaleTimeString();
}

function log(message) {
    console.log(`[${timestamp()}] ${message}`);
}

/**
 * Maps audio slice duration in seconds to the closest LTX frame dropdown label.
 */
function durationToDropdown(durationSec) {
    if (!durationSec || durationSec <= 2.5) return "2 Seconds (49 frames)";
    if (durationSec <= 3.5) return "3 Seconds (73 frames)";
    if (durationSec <= 6.0) return "5 Seconds (121 frames)";
    if (durationSec <= 8.5) return "8 Seconds (193 frames)";
    if (durationSec <= 12.0) return "10 Seconds (241 frames)";
    if (durationSec <= 17.0) return "15 Seconds (361 frames)";
    return "20 Seconds (481 frames)";
}

/**
 * Locate manifest.json in the target directory or path.
 */
function resolveManifest(targetPath) {
    if (!targetPath) return null;
    const resolved = path.resolve(targetPath);
    if (!fs.existsSync(resolved)) return null;

    const stat = fs.statSync(resolved);
    if (stat.isFile()) {
        return {
            manifestPath: resolved,
            folderPath: path.dirname(resolved),
        };
    }
    if (stat.isDirectory()) {
        const manifestCandidate = path.join(resolved, "manifest.json");
        if (fs.existsSync(manifestCandidate)) {
            return {
                manifestPath: manifestCandidate,
                folderPath: resolved,
            };
        }
    }
    return null;
}

/**
 * Health check on backend service.
 */
async function checkBackendHealth(backendUrl) {
    log(`Checking backend connection at ${backendUrl} ...`);
    try {
        const res = await fetch(`${backendUrl}/api/health`);
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
        log(`⚠️ Could not reach backend at ${backendUrl}: ${err.message}`);
        log(`Make sure backend.py is running and the URL is reachable.`);
        return false;
    }
}

/**
 * Uploads media assets and submits the piece to backend.py.
 */
async function uploadAndLaunchJob(backendUrl, pieceData, config) {
    const { audioPath, imagePath, prompt, duration, resolution, aspectRatio, matchAudioDuration, seed, pieceIndex, totalPieces } = pieceData;

    if (!fs.existsSync(audioPath)) {
        throw new Error(`Audio file not found: ${audioPath}`);
    }

    log(`Reading assets for Piece ${pieceIndex}/${totalPieces}...`);
    const audioBuffer = fs.readFileSync(audioPath);
    const audioBlob = new Blob([audioBuffer], { type: "audio/wav" });

    const formData = new FormData();
    formData.append("prompt", prompt);
    formData.append("seed", String(seed ?? -1));
    formData.append("duration", duration ?? "10 Seconds (241 frames)");
    formData.append("resolution", resolution ?? "720p");
    formData.append("aspect_ratio", aspectRatio ?? "Auto (match reference image)");
    formData.append("match_audio_dur", String(matchAudioDuration ?? true));
    formData.append("audio_file", audioBlob, path.basename(audioPath));

    if (imagePath && fs.existsSync(imagePath)) {
        const imageBuffer = fs.readFileSync(imagePath);
        const ext = path.extname(imagePath).toLowerCase();
        const mime = ext === ".webp" ? "image/webp" : (ext === ".jpg" || ext === ".jpeg" ? "image/jpeg" : "image/png");
        const imageBlob = new Blob([imageBuffer], { type: mime });
        formData.append("image_file", imageBlob, path.basename(imagePath));
    } else {
        log(`  Notice: No attached image for Piece ${pieceIndex}, generating text-to-video with audio conditioning.`);
    }

    log(`Submitting Piece ${pieceIndex} to ${backendUrl}/api/generate/upload ...`);
    const response = await fetch(`${backendUrl}/api/generate/upload`, {
        method: "POST",
        body: formData,
    });

    if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Upload failed (HTTP ${response.status}): ${errorText}`);
    }

    const result = await response.json();
    log(`Piece ${pieceIndex} queued with Job ID: ${result.job_id} (Status: ${result.status})`);
    return result.job_id;
}

/**
 * Polls job status until complete.
 */
async function pollJobStatus(backendUrl, jobId, pieceIndex, totalPieces) {
    let lastProgress = -1;
    let lastDesc = "";
    let lastStatus = "";

    while (true) {
        try {
            const res = await fetch(`${backendUrl}/api/jobs/${jobId}`);
            if (!res.ok) {
                throw new Error(`Failed to fetch status: HTTP ${res.status}`);
            }

            const job = await res.json();
            const currentStatus = job.status;
            const progress = job.progress || 0;
            const desc = job.progress_desc || "";

            if (currentStatus !== lastStatus) {
                log(`Piece ${pieceIndex}/${totalPieces} [${jobId}] Status -> ${currentStatus.toUpperCase()}`);
                lastStatus = currentStatus;
            }

            if (progress !== lastProgress || desc !== lastDesc) {
                const pct = Math.round(progress * 100);
                const descStr = desc ? ` (${desc})` : "";
                log(`Piece ${pieceIndex}/${totalPieces} Progress: ${pct}%${descStr}`);
                lastProgress = progress;
                lastDesc = desc;
            }

            if (currentStatus === "completed") {
                const elapsed = job.execution_time ? `${job.execution_time.toFixed(1)}s` : "completed";
                log(`✅ Piece ${pieceIndex}/${totalPieces} completed in ${elapsed}!`);
                if (job.result_video_path) {
                    log(`  Backend output: ${job.result_video_path}`);
                }
                if (job.video_url) {
                    log(`  Video URL: ${backendUrl}${job.video_url}`);
                }
                return job;
            }

            if (currentStatus === "failed") {
                throw new Error(`Job execution failed: ${job.error || "Unknown error"}`);
            }

            if (currentStatus === "cancelled") {
                throw new Error("Job was cancelled by user.");
            }

        } catch (err) {
            if (!err.message.includes("failed:") && !err.message.includes("cancelled")) {
                log(`⚠️ Polling notice: ${err.message}`);
            } else {
                throw err;
            }
        }

        await sleep(POLL_INTERVAL_MS);
    }
}

/**
 * Downloads generated video from backend to local directory.
 */
async function downloadGeneratedVideo(videoUrl, destinationPath) {
    log(`Downloading video to: ${destinationPath} ...`);
    const res = await fetch(videoUrl);
    if (!res.ok) {
        throw new Error(`Failed to download video from ${videoUrl} (HTTP ${res.status})`);
    }
    const arrayBuffer = await res.arrayBuffer();
    fs.writeFileSync(destinationPath, Buffer.from(arrayBuffer));
    const sizeMb = (fs.statSync(destinationPath).size / 1024 ** 2).toFixed(2);
    log(`💾 Saved: ${path.basename(destinationPath)} (${sizeMb} MB)`);
}

async function main() {
    const config = parseCliArgs();

    console.log("========================================");
    console.log("🎬 LTX Video Generator - Batch Manifest Runner");
    console.log(`📡 Backend URL: ${config.backendUrl}`);
    console.log(`📂 Target Folder: ${config.folder}`);
    console.log("========================================\n");

    // Locate and load manifest.json
    const manifestInfo = resolveManifest(config.folder);
    if (!manifestInfo) {
        console.error(`❌ Error: manifest.json not found in "${config.folder}".`);
        console.error(`Please provide a valid folder path containing manifest.json.`);
        console.error(`Example: node auto/check_backend.js "X:\\inputmixed\\Copper Thunder_split_pieces_27_parts (1)"`);
        process.exitCode = 1;
        return;
    }

    const { manifestPath, folderPath } = manifestInfo;
    log(`Loading manifest: ${manifestPath}`);
    const manifestContent = JSON.parse(fs.readFileSync(manifestPath, "utf-8"));

    const projectName = manifestContent.projectName || path.basename(folderPath);
    const allPieces = manifestContent.pieces || [];

    if (allPieces.length === 0) {
        console.error(`❌ Manifest contains no pieces.`);
        process.exitCode = 1;
        return;
    }

    // Filter pieces based on index arguments
    let targetPieces = allPieces;
    if (config.singlePiece !== null) {
        targetPieces = allPieces.filter(p => p.index === config.singlePiece);
    } else {
        if (config.startIndex) {
            targetPieces = targetPieces.filter(p => p.index >= config.startIndex);
        }
        if (config.endIndex) {
            targetPieces = targetPieces.filter(p => p.index <= config.endIndex);
        }
    }

    console.log(`📁 Project Name: ${projectName}`);
    console.log(`📋 Total Pieces in Manifest: ${allPieces.length}`);
    console.log(`🎯 Pieces to Process: ${targetPieces.length} (Index ${targetPieces[0]?.index || 0} to ${targetPieces[targetPieces.length - 1]?.index || 0})`);
    console.log(`📐 Resolution: ${config.resolution} | Match Audio Duration: ${config.matchAudioDuration}`);
    console.log("----------------------------------------\n");

    // Prepare local download directory
    const outputVideosDir = path.join(folderPath, "output_videos");
    if (config.downloadVideos) {
        fs.makedirs = fs.mkdirSync(outputVideosDir, { recursive: true });
    }

    // Health check
    const isHealthy = await checkBackendHealth(config.backendUrl);
    if (!isHealthy) {
        console.error("\n❌ Exiting: Backend service is not reachable.");
        process.exitCode = 1;
        return;
    }

    let completed = 0;
    let failed = 0;
    let skipped = 0;

    for (let i = 0; i < targetPieces.length; i++) {
        const piece = targetPieces[i];
        const pieceIndex = piece.index;
        const totalPieces = allPieces.length;

        // Destination video file
        const videoOutputName = piece.videoFileName
            ? piece.videoFileName.replace(/\.[^.]+$/, ".mp4")
            : `${projectName}_part${String(pieceIndex).padStart(2, "0")}.mp4`;
        const localVideoPath = path.join(outputVideosDir, videoOutputName);

        // Check if already completed and exists locally
        if (config.skipExisting && config.downloadVideos && fs.existsSync(localVideoPath) && fs.statSync(localVideoPath).size > 1000) {
            log(`⏭️ Piece ${pieceIndex}/${totalPieces} already exists locally: ${videoOutputName} (${(fs.statSync(localVideoPath).size / 1024 ** 2).toFixed(1)} MB) - Skipping!`);
            skipped++;
            continue;
        }

        const audioPath = path.join(folderPath, piece.audioFileName);
        const imagePath = (piece.hasAttachedImage && piece.imageFileName) ? path.join(folderPath, piece.imageFileName) : null;
        const durationDropdown = durationToDropdown(piece.duration);

        console.log("\n========================================");
        log(`Processing Piece ${pieceIndex}/${totalPieces} (${piece.start}s -> ${piece.end}s, dur: ${piece.duration}s)`);
        log(`Prompt: ${ piece.prompt}`);
        log(`🎵 Audio: ${path.basename(audioPath)}`);
        if (imagePath) log(`🖼️ Image: ${path.basename(imagePath)}`);
        log(`⏱️ Duration Dropdown: ${durationDropdown} | Snap Audio: true`);
        console.log("========================================");

        try {
            const pieceData = {
                audioPath,
                imagePath,
                prompt: piece.prompt,
                duration: durationDropdown,
                resolution: config.resolution,
                aspectRatio: config.aspectRatio,
                matchAudioDuration: config.matchAudioDuration,
                seed: config.seed,
                pieceIndex,
                totalPieces,
            };

            const jobId = await uploadAndLaunchJob(config.backendUrl, pieceData, config);
            const finishedJob = await pollJobStatus(config.backendUrl, jobId, pieceIndex, totalPieces);

            // Download video locally
            if (config.downloadVideos && finishedJob.video_url) {
                const fullVideoUrl = `${config.backendUrl}${finishedJob.video_url}`;
                try {
                    await downloadGeneratedVideo(fullVideoUrl, localVideoPath);
                } catch (dlErr) {
                    log(`⚠️ Warning: Could not download video locally: ${dlErr.message}`);
                }
            }

            completed++;

        } catch (error) {
            failed++;
            console.error(`\n[${timestamp()}] ❌ Piece ${pieceIndex} failed:`, error.message || error);
        }

        log(`Batch Progress: ${i + 1}/${targetPieces.length} pieces processed (${completed} succeeded, ${skipped} skipped, ${failed} failed).`);
    }

    console.log("\n========================================");
    log(`🎉 BATCH EXECUTION COMPLETE`);
    log(`Project: ${projectName}`);
    log(`Total: ${targetPieces.length} | Succeeded: ${completed} | Skipped: ${skipped} | Failed: ${failed}`);
    if (config.downloadVideos) {
        log(`📁 Local Videos Saved in: ${outputVideosDir}`);
    }
    console.log("========================================");
}

main().catch(error => {
    console.error("Fatal error:", error);
    process.exitCode = 1;
});
