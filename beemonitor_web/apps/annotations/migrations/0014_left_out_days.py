from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("annotations", "0013_reviewers_not_annotators"),
    ]

    operations = [
        migrations.AddField(
            model_name="annotationproject",
            name="left_out_days",
            field=models.TextField(blank=True, default=""),
        ),
    ]
