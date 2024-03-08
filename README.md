Run program with `--release` flag for better performance:
```
cargo run --release
```

`ffmpeg` command to run to transform the audio file to an input file accpeted by yamnet:
```
ffmpeg -i [audiofilename].[ext] -acodec pcm_f32le -ar 16000 -ac 1 -f wav [audiofilename].wav
````
