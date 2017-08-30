import boto
from boto.s3.key import Key

def upload_to_s3(bucketname,filename,directory,AWS_KEY,AWS_SECRET_KEY):
    """ Upload to S3:
    bucketname: name of the bucket to upload to
    filenam: name of the file
    directory: directory of the file currently on your local machine
    AWS_KEY: Your AWS Key
    AWS_SECRET_KEY: Your AWS Secret Key
    """
    
    if not directory.endswith('/'):
        directory = directory + '/'

    conn = boto.connect_s3(AWS_KEY,
          AWS_SECRET_KEY)

    bucket = conn.get_bucket(bucketname)

    # print ('Uploading %s to Amazon S3 bucket %s') % (filename, bucket_name)

    k = Key(bucket)
    k.key = filename
    k.set_contents_from_filename(directory+filename,num_cb=10)
    print("File Uploaded: ",filename)
    
