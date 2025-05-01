import argparse
import platform
import sounddevice as sd
import soundfile as sf
import time
import tkinter as tk

from tkinter import messagebox
from pylsl import StreamInfo, StreamOutlet, local_clock

from pathlib import Path
import os
import random


class MusicExperiment:
    def __init__(self, session_number):
        # Paths configuration

        self.path_to_session = f"../../data/music_sessions/session{session_number}"
        
        # Initialize Tkinter
        self.root = tk.Tk()
        self.root.title("Music Experiment")
        self.root.geometry("800x600")
        self.label = tk.Label(self.root, text="", font=("Arial", 24))
        self.label.pack(expand=True)
        
        # Space key handling
        self.space_pressed = tk.BooleanVar()
        self.root.bind("<KeyPress>", self.on_key)
        
        # Initialize LSL streams
        self._init_lsl_streams()
        
        # Load songs
        self.song_sequence = self._load_songs(path_to_session=self.path_to_session)

    def _init_lsl_streams(self):
        """Initialize LSL streams for markers and song info"""
        try:
            # Marker stream
            self.stream_info_marker = StreamInfo(
                name='MusicMarkers',
                type='Markers',
                channel_count=1,
                nominal_srate=0,
                channel_format='string',
                source_id=platform.node()
            )
            print(f"LSL Stream created: {self.stream_info_marker.name()} "
                  f"{self.stream_info_marker.type()} {self.stream_info_marker.source_id()}")
            self.marker_outlet = StreamOutlet(self.stream_info_marker)

            # Song info stream
            self.stream_info_song = StreamInfo(
                name='Song',
                type='Info',
                channel_count=2,
                nominal_srate=0,
                channel_format='string',
                source_id=platform.node()
            )
            print(f"LSL Stream created: {self.stream_info_song.name()} "
                  f"{self.stream_info_song.type()} {self.stream_info_song.source_id()}")
            self.song_outlet = StreamOutlet(self.stream_info_song)
        except Exception as e:
            print(f"Error initializing LSL streams: {e}")
            raise

    def _load_songs(self, path_to_session):
        """Load and prepare song sequence"""
        try:
            # get the names of the song files
            audio_files = []
            for song in os.listdir(path_to_session):
                audio_files.extend(song)
            
            random.shuffle(audio_files)

            return audio_files
        
        except Exception as e:
            print(f"Error loading songs: {e}")
            return []

    def show_text(self, message, duration=None):
        """Display text on the screen"""
        self.label.config(text=message)
        self.root.update()
        if duration:
            time.sleep(duration)

    def wait_for_keypress(self):
        """Wait for space key press"""
        self.root.wait_variable(self.space_pressed)

    def on_key(self, event):
        """Handle key press events"""
        if event.keysym == "space":
            self.space_pressed.set(True)

    def split_song(self, song: str):
        return song.split(".")

    def run(self):
        """Run the experiment"""
        try:
            # Initial screen
            self.show_text("Ready? Press SPACE to continue")
            self.wait_for_keypress()

            # Silence period
            # TODO change to 5s
            self.show_text("5 seconds silence", duration=5)

            # Main song loop
    
            song = 0
            for song in self.song_sequence:
                print(f"Song {song} out of {len(self.song_sequence)}")
                print("Pushing marker:", 'Start Recording')
                self.marker_outlet.push_sample(['Start Recording'], local_clock())

                [genre, song_id, extension] = song.split(".")

                # Change information that is pushed to the stream
                print(f"Pushing song: {genre + song_id}")
                self.song_outlet.push_sample([f'{song_id}', f'{genre}'], local_clock())
                
                self.show_text("Listen to the song")
                
                # Play song
                data, samplerate = sf.read(song) # TODO Adapt to actually have the path to the audio path.
                sd.play(data, samplerate)
                sd.wait()
                
                print("Pushing marker:", 'End Recording')
                self.marker_outlet.push_sample(['End Recording'], local_clock())
                
                # TODO lower the duration
                self.show_text("Wait for next song!", duration=5)

                song += 1

            # End screen
            self.show_text("Thank you for participating!", duration=5)
            messagebox.showinfo("Experiment", "Thank you for participating!")
            
        except Exception as e:
            print(f"Error during experiment: {e}")
            messagebox.showerror("Error", f"An error occurred: {e}")
        finally:
            self.root.destroy()


def main():
    """Main function to start the experiment"""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run a music experiment with specified songs and genres')
    parser.add_argument('--session', type=int, default=1, help='Number of the session')
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        experiment = MusicExperiment(session_number=args.session)
        experiment.run()
    except Exception as e:
        print(f"Failed to start experiment: {e}")

if __name__ == "__main__":
    main()