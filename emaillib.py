"""Use the local machine's Outlook instance and configured user account to send emails. Windows only.

Outlook must be installed and configured for the currently logged in
account under which python is executing.
"""

# Had to rename from email.py because of name clashes
import os.path as _path
import time as _time

import funclite.stringslib as _stringslib
import funclite.errors as _errors


class EmailJobStatus:
    """
    Handles generating standard emails, aimed at automated emails generated for file transfer or reporting operatings.

    Args:
        job_name (str): Name of the job (e.g. 'Sharepoint Synchronisation'
        subject: Email subject, leave blank to create from the job name
        body: Text to include in the body of the email, between body_header and body_footer
        body_header: First text to appear in the body, leave blank to use the job name and current date-time
        body_footer: Last text to appear in the foot, leave blank to default to 'This email was automatically generated ...'
        files_save_dir: Complete if want to include message where attachments were saved to

    Notes:
        body set on instance instantiation is appended to header, footer and error statuses. The body property concatenated these
        values together and is the body intended to be set for the email. See example.

    Examples:
        >>> Obj = EmailJobStatus('Sharepoint Sync', body='This is notification of the status of sharepoint sync from sp.py', files_save_dir='c:\')
        >>> Obj.add_success('Synched from here to there')
        >>> Obj.add_err('Sync to there failed')
        >>> E = Email(to='my@home', subject=Obj.subject, body=Obj.body, attachments=['c:/my.doc'])
        >>> E.send()
    """

    def __init__(self, job_name, subject='', body='', body_header='', body_footer='This email was automatically generated. Email the sender to be removed from the distribution list.',
                 files_save_dir=''):
        self._job_name = job_name
        self._errors = list()
        self._body = body
        self._successes = list()
        self._body_footer = body_footer
        if body_header:
            self._body_header = body_header
        else:
            self._body_header = '%s at %s' % (job_name, _stringslib.pretty_date_time_now())

        if subject:
            self.subject = subject
        else:
            self.subject = '%s at %s' % (job_name, _stringslib.pretty_date_time_now())

        if files_save_dir:
            self.files_save_dir = 'File(s) were saved to "%s"' % files_save_dir
        else:
            self.files_save_dir = None

    def add_err(self, err: str):
        """
        Add an error

        Args:
            err: error string

        Returns: None
        """
        self._errors.append(err)

    def add_success(self, success: str):
        """
        Add a success

        Args:
            success (str): Add a success

        Returns: None

        """
        self._successes.append(success)

    @property
    def body(self):
        bd = [self._body_header + '\n']
        if self._body:
            bd.append(self._body)

        if self._successes:
            bd.append('Successes\n==========')
            bd.append('\n'.join(self._successes))

        if self._errors:
            bd.append('\nErrors\n==========')
            bd.append('\n'.join(self._errors))

        if self.files_save_dir:
            bd.append('\n\nAttachments were saved to %s' % self.files_save_dir)

        if self._body_footer:
            bd.append('\n\n%s' % self._body_footer)

        s = '\n'.join(bd)
        return s


class Email:
    """
    Send an email using the users outlook account (this will be the user in which the
    python context is executing)

    Args:
        to (str): email list, e.g. g@home;me@west;you@hotmail.com
        subject (str): subject text
        body (str): body text
        attachments (list,tuple,str,None): file paths to add a attachments
        
    Notes:
        Use an instance of emails.EmailJobStatus to handle generating an email's subject and body
        
    Examples:
        Create an EmailJobStatus instance to handle subject and body creation\n
        >>> Obj = EmailJobStatus('Sharepoint Sync', body='This is notification of the status of sharepoint sync from sp.py', files_save_dir='c:\')
        >>> Obj.add_success('Synched from here to there')
        >>> Obj.add_err('Sync to there failed'
        
        Send an email to a list of known recipients\n
        >>> E = Email(to=Email.concat_emails('me@here.com', 'you@there.com'), subject=Obj.subject, body=Obj.body, attachments=['c:/my.doc'])
        >>> E.send()
    """

    def __init__(self, to: str, subject: str, body: str, attachments: (list, tuple, str) = ()):
        self.to = to
        self.subject = subject
        self.body = body
        self._outlook = None
        self.attachments = []
        if attachments:
            if isinstance(attachments, str):
                attachments = [attachments]
            self.attachments = list(map(_path.normpath, attachments))
        self._setapp()

    def _setapp(self):
        try:
            from win32com import client
            app = client.gencache.EnsureDispatch('outlook.application')
        except AttributeError:
            # Corner case dependencies.
            import os
            import re
            import sys
            import shutil
            # Remove cache and try again.
            MODULE_LIST = [m.__name__ for m in sys.modules.values()]
            for module in MODULE_LIST:
                if re.match(r'win32com\.gen_py\..+', module):
                    del sys.modules[module]
            shutil.rmtree(os.path.join(os.environ.get('LOCALAPPDATA'), 'Temp', 'gen_py'))
            from win32com import client
            app = client.gencache.EnsureDispatch('outlook.application')
        if app:
            self._outlook = app
        else:
            raise _errors.EmailNoOutlookApp('Failed to get an instance of the Outlook app. Is it installed?')

    def send(self) -> None:
        """Send the email

        Raises:
            ValueError: If Email.to is empty.

        Returns: None
        """
        if not self.to:
            raise ValueError('Email.send was called, but no recipient was set. Nothing sent.')

        mail = self._outlook.CreateItem(0)
        mail.To = self.to
        mail.Subject = self.subject
        mail.Body = self.body
        for s in self.attachments:
            mail.Attachments.Add(Source=s, Type=1)
        mail.Send()

    @staticmethod
    def pretty_now():
        return _time.strftime('%H:%M%p %Z on %b %d, %Y')

    @staticmethod
    def concat_emails(*args) -> str:
        """concatenate emails into a str

        Args:
            *args iter(str): email addresses

        Examples:
            >>> concat_emails('c@home.com','me@wales.gov')  # noqa
            'c@home.com;me@wales.gov'
        """
        return ';'.join(args)
