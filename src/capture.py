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
            
            # Calculate DD Flight (Time since last key PRESS)
            dd_flight = 0.0
            if self.current_record:
                # Get the press time of the previous key (which is the last one in current_record)
                # Note: current_record stores processed keys. The 'last' key pressed might not be fully processed 
                # if it hasn't been released yet. 
                # HOWEVER, for typing streams, we usually care about the sequence.
                # Let's rely on the last APPENDED record for the "previous key".
                # This assumes sequential typing (which is true for the password phrase).
                last_press = self.current_record[-1]['press_ts']
                dd_flight = now - last_press

            # We store the press time temporarily to be added to the record on release
            # Or we can track pending presses. 
            # But since we build the record ON RELEASE (to get dwell time), we need to pass this DD info forward.
            # A cleaner way: Store 'last_global_press_time' in the class.
            
            # actually, let's just wait for release to package everything.

        elif event.event_type == 'up':
            if k in self.press_times:
                start_time = self.press_times.pop(k)
                dwell = now - start_time # This is 'Hold' time

                ud_flight = 0.0
                dd_flight = 0.0
                
                if self.current_record:
                    last_release = self.current_record[-1]['release_ts']
                    last_press = self.current_record[-1]['press_ts']
                    
                    ud_flight = start_time - last_release
                    dd_flight = start_time - last_press

                self.current_record.append({
                    'key': k,
                    'hold': dwell,
                    'ud': ud_flight,
                    'dd': dd_flight,
                    'press_ts': start_time,
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
