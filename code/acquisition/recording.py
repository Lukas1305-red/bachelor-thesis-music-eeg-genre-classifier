import argparse
import platform
import sounddevice as sd
import soundfile as sf
import time
import tkinter as tk

from tkinter import messagebox
from pylsl import StreamInfo, StreamOutlet, local_clock

from utils import load, get_audio_path


class MusicExperiment:
    def __init__(self, n_songs, genres, session_number):
        # Paths configuration

        # TODO Adapt the path for my dataset

        if platform.system() == 'Windows':
            self.path_to_fma_small = "C:/Users/Mitarbeiter/Music/data/fma_small"
            self.path_to_fma_metadata = "C:/Users/Mitarbeiter/Music/data/fma_metadata"
        else:
            self.path_to_fma_small = "/home/pistrosch/Musik/fma_small"
            self.path_to_fma_metadata = "/home/pistrosch/Musik/fma_metadata"
        
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
        self.song_sequence, self.labels = self._load_songs(n_songs, genres, session_number)

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

    # TODO Adapt this function for my purposes
    def _load_songs(self, n_songs, genres, session_number):
        """Load and prepare song sequence"""
        try:
            tracks = load(self.path_to_fma_metadata + '/tracks.csv')
            small = tracks[tracks['set', 'subset'] <= 'small']

            small = small[small['track']['genre_top'].isin(genres)]

            small = small['track'][['genre_top']]

            # Search for music files
            # TODO Adapt or remove this part since i already do the session generatoin in a different notebook
            # TODO maybe just shuffle the data after loading --> Probably just load the names in and take a random and play that
            songs_per_genre = n_songs // len(genres)
            hop = 1000 // songs_per_genre

            if session_number >= hop:
                raise ValueError('Session-ID is too high! You likely record music that you already recorded!')

            ids = [i + (session_number - 1) for i in range(0, 1000, hop)]
            print(f'Songs per Genre: {len(ids)}')

            all_songs = []
            for genre in genres:
                df_genre = small[small['genre_top'] == genre]
                df_genre = df_genre.iloc[ids]
                df_genre = df_genre['genre_top']

                all_songs.append(df_genre)

            # Create alternating sequence
            song_sequence = []
            labels = []
            for i in range(0, len(ids) // 5):
                for genre_songs in all_songs:
                    songs = genre_songs.iloc[i * 5: (i + 1) * 5]
                    song_sequence.extend(list(songs.index))
                    labels.extend(list(songs.values))

            print(f"Loaded {len(song_sequence)} songs")
            return song_sequence, labels
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

    def run(self):
        """Run the experiment"""
        try:
            # Initial screen
            self.show_text("Ready? Press SPACE to continue")
            self.wait_for_keypress()

            # Silence period
            # TODO change to 5s
            self.show_text("10 seconds silence", duration=10)

            # Main song loop
            # TODO Adapt loop condition
            song = 0
            for song_number, label in zip(self.song_sequence, self.labels):
                print(f"Song {song} out of {len(self.song_sequence)}")
                print("Pushing marker:", 'Start Recording')
                self.marker_outlet.push_sample(['Start Recording'], local_clock())

                # Change information that is pushed to the stream
                print(f"Pushing song: {song_number}")
                self.song_outlet.push_sample([f'{song_number}', f'{label}'], local_clock())
                
                self.show_text("Listen to the song")
                
                # Play song
                data, samplerate = sf.read(get_audio_path(self.path_to_fma_small, song_number))
                sd.play(data, samplerate)
                sd.wait()
                
                print("Pushing marker:", 'End Recording')
                self.marker_outlet.push_sample(['End Recording'], local_clock())
                
                # TODO lower the duration
                self.show_text("Wait for next song!", duration=10)

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
    parser.add_argument('--n_songs', type=int, default=100, help='Number of songs for the experiment')
    parser.add_argument('--genres', nargs='+', default=['Instrumental', 'Rock'], 
                       help='List of genres for the experiment (space-separated)')
    parser.add_argument('--session', type=int, default=1, help='Number of the session')
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        experiment = MusicExperiment(n_songs=args.n_songs, genres=args.genres, session_number=args.session)
        experiment.run()
    except Exception as e:
        print(f"Failed to start experiment: {e}")

if __name__ == "__main__":
    main()