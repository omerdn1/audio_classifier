// `ffmpeg` command to run to transform the audio file to an input file accpeted by yamnet
// ffmpeg -i white.wav -acodec pcm_f32le -ar 16000 -ac 1 -f wav white.wav

use std::collections::HashMap;
use std::io::BufRead;
use std::path::Path;
use tract_onnx::prelude::*;
use tract_onnx::tract_hir::tract_ndarray::Axis;

fn main() -> TractResult<()> {
    // Load the ONNX model
    let model_path = Path::new("./models/yamnet.onnx");
    let model = tract_onnx::onnx()
        .model_for_path(model_path)?
        .with_input_fact(
            0,
            InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 1, 96, 64)),
        )?
        .into_optimized()?
        .into_runnable()?;

    // Read and preprocess the audio file
    let audio_data = preprocess_audio("/Users/omerdangoor/Downloads/blade.wav")?;

    // Run the model
    let result = model.run(tvec!(audio_data.into()))?;

    // Read labels from file
    let labels = read_labels("./models/yamnet_label_list.txt")?;

    // Analyze the output
    let top_labels = analyze_output(
        result
            .into_iter()
            .map(|value| value.into_tensor())
            .collect(),
        &labels,
    )?;

    println!("Top 5 labels:");
    for label in top_labels.iter() {
        println!("{}", label);
    }
    Ok(())
}

fn preprocess_audio(audio_path: &str) -> TractResult<Tensor> {
    // Read the audio file
    let mut reader = hound::WavReader::open(audio_path).expect("Cannot read audio file");
    let samples: Vec<f32> = reader
        .samples::<i16>()
        .filter_map(Result::ok)
        .map(|s| s as f32 / i16::MAX as f32) // Normalize to [-1.0, +1.0]
        .collect();

    // Handle stereo to mono conversion if needed
    let mono_samples = if reader.spec().channels == 2 {
        stereo_to_mono(&samples)
    } else {
        samples
    };

    // Reshape the audio data to the expected input shape [1, 1, 96, 64]
    // This is a placeholder; adjust the reshaping logic based on how the model expects the data
    let reshaped_audio_data = reshape_audio_data(mono_samples, 96, 64)?;

    Ok(reshaped_audio_data)
}

fn stereo_to_mono(samples: &[f32]) -> Vec<f32> {
    samples
        .chunks(2)
        .map(|chunk| (chunk[0] + chunk[1]) / 2.0)
        .collect()
}

fn reshape_audio_data(samples: Vec<f32>, height: usize, width: usize) -> TractResult<Tensor> {
    // Placeholder function to reshape the audio data
    // Implement this based on the specific requirements of your model
    // The following is a simplistic approach and might not be correct:
    let total_size = height * width;
    let mut reshaped = vec![0.0; total_size];
    let samples_len = samples.len().min(total_size);
    reshaped[..samples_len].copy_from_slice(&samples[..samples_len]);
    tensor1(&reshaped).into_shape(&[1, 1, height, width])
}

fn analyze_output(result: Vec<Tensor>, labels: &[String]) -> Result<Vec<String>, TractError> {
    let output = result[0].to_array_view::<f32>()?;

    // Collect scores with their respective indices
    let mut scores_with_indices: Vec<(usize, f32)> = output
        .index_axis(Axis(0), 0)
        .into_iter()
        .enumerate()
        .map(|(index, &score)| (index, score))
        .collect();

    // Sort by score in descending order
    scores_with_indices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Get top 5 scores (or less if there aren't enough)
    let top_scores = scores_with_indices
        .iter()
        .take(5)
        .map(|&(index, _)| {
            labels
                .get(index)
                .cloned()
                .unwrap_or_else(|| "Unknown label".to_string())
        })
        .collect::<Vec<_>>();

    Ok(top_scores)
}

fn read_labels(labels_path: &str) -> Result<Vec<String>, std::io::Error> {
    let file = std::fs::File::open(labels_path)?;
    let reader = std::io::BufReader::new(file);
    reader.lines().collect()
}
