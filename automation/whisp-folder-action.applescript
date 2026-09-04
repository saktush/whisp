-- Folder Action attached to the whisp repo folder: fires whenever items are added to
-- it. Forwards each added item to process-new-file.sh, one at a time, in
-- the order Finder reports them. That script filters for audio/video types,
-- waits for the file to finish being written, and uses a lock so at most one
-- transcription ever runs at once -- even if this handler fires again for a
-- second batch of files while the first is still processing.
property processScript : "__PROCESS_SCRIPT__"

on adding folder items to thisFolder after receiving addedItems
	repeat with anItem in addedItems
		set itemPath to POSIX path of (anItem as alias)
		try
			with timeout of 86400 seconds
				do shell script quoted form of processScript & " " & quoted form of itemPath
			end timeout
		on error errMsg
			try
				display notification errMsg with title "whisp automation error"
			end try
		end try
	end repeat
end adding folder items to
