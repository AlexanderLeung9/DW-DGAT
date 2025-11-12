Option Explicit

Dim tShell
Set tShell = CreateObject("WScript.Shell")

Dim tFSO
Set tFSO = CreateObject("Scripting.FileSystemObject")

If WScript.Arguments.Length <> 3 Then
	WScript.Echo "Invalid argument number! (" & WScript.Arguments.Length & ")"
	WScript.Quit 1
End If

Dim tWaitSeconds, tSourceFolderPath, tDestinationFolderPath
tWaitSeconds = CInt(WScript.Arguments.Item(0))
tSourceFolderPath = WScript.Arguments.Item(1)
tDestinationFolderPath = WScript.Arguments.Item(2)

WScript.Sleep tWaitSeconds * 1000

If tFSO.FolderExists(tSourceFolderPath) Then
	tFSO.MoveFolder tSourceFolderPath, tDestinationFolderPath
Else
	MsgBox """" & tSourceFolderPath & """" & " doesn't exist!", vbOKOnly + vbCritical, WScript.ScriptName
End If
