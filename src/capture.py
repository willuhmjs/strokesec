"""
Shared keystroke capturing logic for the Keystroke Dynamics Authentication System.
"""
import time
import keyboard

class KeystrokeCapture:
    """
    Captures keystroke dynamics (hold time, UD flight, DD flight).
    """
    def __init__(self):
        self.press_times = {}
        self.raw_key_events = [] # Stores {key, press_ts, release_ts}
        self.done = False

    def on_key_event(self, event):
        """
        Callback for keyboard events.
        """
        k = event.name
        now = event.time

        if event.event_type == 'down':
            if k not in self.press_times:
                self.press_times[k] = now
            
        elif event.event_type == 'up':
            if k in self.press_times:
                start_time = self.press_times.pop(k)
                
                # Store raw event data
                self.raw_key_events.append({
                    'key': k,
                    'press_ts': start_time,
                    'release_ts': now
                })

                if k == 'enter':
                    self.done = True

    def process_sequence(self):
        """
        Sorts raw events by press time and calculates flight metrics.
        """
        # 1. Sort by press timestamp to handle rollover typing correctly
        sorted_events = sorted(self.raw_key_events, key=lambda x: x['press_ts'])
        
        processed_record = []
        
        for i, event in enumerate(sorted_events):
            key = event['key']
            press_ts = event['press_ts']
            release_ts = event['release_ts']
            
            # Calculate Hold Time (Dwell)
            hold_time = release_ts - press_ts
            
            ud_flight = 0.0
            dd_flight = 0.0
            
            if i > 0:
                prev_event = sorted_events[i-1]
                prev_release = prev_event['release_ts']
                prev_press = prev_event['press_ts']
                
                # Calculate Flight Times
                ud_flight = press_ts - prev_release
                dd_flight = press_ts - prev_press

            processed_record.append({
                'key': key,
                'hold': hold_time,
                'ud': ud_flight,
                'dd': dd_flight,
                'press_ts': press_ts,
                'release_ts': release_ts
            })
            
        return processed_record

    def capture_sequence(self):
        """
        Captures a single sequence of keystrokes until ENTER is pressed.
        Returns the list of processed keystroke data.
        """
        self.press_times = {}
        self.raw_key_events = []
        self.done = False
        
        # We'll use a local hook to avoid interference
        hook = keyboard.hook(self.on_key_event)
        
        try:
            while not self.done:
                time.sleep(0.01)
        finally:
            keyboard.unhook(hook)
            
        # Post-process the sequence
        return self.process_sequence()
