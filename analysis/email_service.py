import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders


def emailOTPtoUser(targetAddress, message, file_paths):
    serviceAddress = "hyperband8@gmail.com"
    servicePassword = "Hyperband1"

    msg = MIMEMultipart()
    msg['From'] = serviceAddress
    msg['To'] = targetAddress
    msg['Subject'] = "Experiment completed!"

    body = message
    msg.attach(MIMEText(body, 'plain'))

    for file_path in file_paths:
        file_name = file_path.split("/")[-1]

        # open the file to be sent
        attachment = open(file_path, "rb")

        # instance of MIMEBase and named as p
        p = MIMEBase('application', 'octet-stream')

        # To change the payload into encoded form
        p.set_payload((attachment).read())

        # encode into base64
        encoders.encode_base64(p)

        p.add_header('Content-Disposition', "attachment; filename= %s" % file_name)

        # attach the instance 'p' to instance 'msg'
        msg.attach(p)

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(serviceAddress, servicePassword)
    text = msg.as_string()
    server.sendmail(serviceAddress, targetAddress, text)

    print(" \n Email sent with message: " + message)
    server.quit()
