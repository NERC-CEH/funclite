class DontRunThisYouIdjit(Exception):
    """Dont run this again you idiot! Used for making script code unexecutable while retaining it as an example"""


# email.py
class EmailNoOutlookApp(Exception):
    """Failed to get an instance of the Outlook app. Is it installed?"""




# Sharepoint
class SharepointUploadFailed(Exception):
    """A sharepoint upload failed"""

class SharepointDownloadFailed(Exception):
    """A sharepoint download failed"""
