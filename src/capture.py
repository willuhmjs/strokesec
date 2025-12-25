"""
Shared keystroke capturing logic for the Keystroke Dynamics Authentication System.
"""
import time
import keyboard

class KeystrokeCapture:
    """
    Captures keystroke dynamics (dwell time and flight time).
    """
    def __init__(self):
        self.press_times = {}
        self.current_record = []
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
                dwell = now - start_time

                flight = 0.0
                if self.current_record:
                    last_release = self.current_record[-1]['release_ts']
                    flight = start_time - last_release

                self.current_record.append({
                    'key': k,
                    'dwell': dwell,
                    'flight': flight,
                    'release_ts': now
                })

                if k == 'enter':
                    self.done = True

    def capture_sequence(self):
        """
        Captures a single sequence of keystrokes until ENTER is pressed.
        Returns the list of keystroke data.
        """
        self.press_times = {}
        self.current_record = []
        self.done = False
        
        # We'll use a local hook to avoid interference if multiple instances run (though unlikely here)
        # But keyboard.hook is global. So we just hook, wait, unhook.
        hook = keyboard.hook(self.on_key_event)
        
        try:
            while not self.done:
                time.sleep(0.01)
        finally:
            keyboard.unhook(hook)
            
        return self.current_record
